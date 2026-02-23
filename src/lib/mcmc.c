/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <phast/misc.h>
#include <phast/trees.h>
#include <nj.h>
#include <likelihoods.h>
#include <gradients.h>
#include <mcmc.h>
#include <multi_mvn.h>
#include <variational.h>
#include <geometry.h>

/* MCMC refinement of variational samples */

List *nj_var_sample_mcmc(int nsamples, int thin, multi_MVN *mmvn,
                         CovarData *data, TreeModel *mod, FILE *logf) {

  int n = data->nseqs, dim = data->dim, fulld = n * dim;
  int niters = 0, naccept = 0, last_naccept = 0, nsamp = 0;
  unsigned int keep_sampling = TRUE, burnin = TRUE, accept = FALSE;
  double rho = 0.99, s = 1.0, accrt = 0.0, new_accrt = 0.0; 
  Vector *mu = vec_new(fulld);
  double nf_logdet;
  mmvn_save_mu(mmvn, mu);
  TreeNode *tree = NULL, *oldtree = NULL;
  List *retval = lst_new_ptr(nsamples);

  if (n >= 250) rho = 0.995; /* more conservative for higher dimensions */
      
  /* initialize last_lnl based on mean; can we do this below by being clever? */
  double lnl = 0.0, lastlnl = 0.0; /* placeholder */

  if (logf != NULL) {
    fprintf(logf, "### Starting MCMC sampling at posterior mean (burn-in of %d iterations, target acceptance rate %.3f)\n",
      BURNIN_ITERS, TARGET_ACCEPT_RATE);
    fprintf(logf, "## %s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "iter", "lnl", "prevlnl", "burnin",
            "accept", "block_accrt", "tot_accrt", "rho", "s");
    
  }

  /* start with mod->tree == NULL and free any previous tree; all
   * persistent tree handling will be with tree and oldtree */
  if (mod->tree != NULL) {
    tr_free(mod->tree);
    mod->tree = NULL;
  }
  
  Vector *lastz = vec_new(fulld), *zprop = vec_new(fulld),
    *zeta = vec_new(fulld), *x = vec_new(fulld), *y = vec_new(fulld);
  vec_zero(lastz);

  while (keep_sampling) {
    niters++;
    burnin = niters <= BURNIN_ITERS;
    
    /* propose new z by preconditioned Crank-Nicolson */
    vec_copy(zprop, lastz);
    vec_scale(zprop, rho);
    mvn_sample_std(zeta);
    vec_scale(zeta, sqrt(1 - rho * rho));
    vec_plus_eq(zprop, zeta);

    /* propose new x centered on variational mean; param s is tuned
     * dynamically (below) */
    vec_copy(x, zprop);
    vec_scale(x, s);
    vec_plus_eq(x, mu); 
    
    /* convert x to y using normalizing flows if available */
    nj_apply_normalizing_flows(y, x, data, &nf_logdet);

    /* set up baseline objects */
    nj_points_to_distances(y, data);
    tree = nj_inf(data->dist, data->names, NULL, NULL, data);
    mod->tree = NULL; /* prevent nj_reset_tree_model from freeing the tree */
    nj_reset_tree_model(mod, tree);

    /* calculate log likelihood and analytical gradient */
    if (data->crispr_mod != NULL)
      lnl = cpr_compute_log_likelihood(data->crispr_mod, NULL);
    else
      lnl = nj_compute_log_likelihood(mod, data, NULL);
    
    /* also get migration log likelihood if needed (skip during warmup) */
    if (data->migtable != NULL &&
      !(data->crispr_mod != NULL && data->crispr_mod->mig_warmup)) {
      lnl += mig_compute_log_likelihood(mod, data->migtable, data->crispr_mod,
        NULL);
    }
    
    double alpha = lastlnl == 0 ? 1 : fmin(1.0, exp(lnl - lastlnl));
    if (unif_rand() < alpha) {
      accept = TRUE;
      lastlnl = lnl;
      vec_copy(lastz, zprop);
      naccept++;
      if (oldtree != NULL) {
        tr_free(oldtree); /* free old tree */
        oldtree = NULL;
      }
    }
    else {
      accept = FALSE;
      tr_free(tree); /* free rejected tree */
      tree = oldtree; /* point to old tree */
      oldtree = NULL; 
    }

    /* now 'tree' contains the current state of the chain, whether we
     * accepted or not; secondary tree has been freed and oldtree == NULL */

    accrt = naccept / (double)niters;

    /* during burnin only, adapt s and rho to target acceptance rate;
     * after burnin, keep fixed */
    if (burnin && niters % TUNING_INTERVAL == 0) {
      int new_naccept =
          naccept - last_naccept; /* number accepted since last check */
      int t = niters / TUNING_INTERVAL; /* number of checks so far */
      new_accrt =
        new_naccept / (double)TUNING_INTERVAL; /* acceptance rate since last check */

      /* diminishing learning rates */
      double eta_rho = 0.1 / sqrt((double)t);   /* fine tuning */
      double eta_s   = 0.2 / sqrt((double)t);   /* coarse tuning */

      /* rho and s are intertwined so we'll update them on alternating blocks */

      if (t % 2 == 0) {
        /* update rho on log scale */
        rho = exp(log(rho) + eta_rho * (new_accrt - TARGET_ACCEPT_RATE));
        if (rho < 0.90)   rho = 0.90;
        if (rho > 0.9995) rho = 0.9995;
      }
      else {
        /* update s on log scale */
        s = exp(log(s) + eta_s * (new_accrt - TARGET_ACCEPT_RATE));
        if (s < 0.1) s = 0.1;
        if (s > 10)  s = 10;
      }

      last_naccept = naccept;
    }

    if (!burnin && niters % thin == 0) {
      if (tree == NULL) {
        /* special case where all iterations since last collect were
         * rejections; reconstruct current state tree from lastz for
         * output */
        vec_copy(x, lastz);
        vec_scale(x, s);
        vec_plus_eq(x, mu);
        nj_apply_normalizing_flows(y, x, data, &nf_logdet);
        nj_points_to_distances(y, data);
        tree = nj_inf(data->dist, data->names, NULL, NULL, data);
        mod->tree = NULL; /* clear dangling ptr left by rejected proposed tree */
      }
      lst_push_ptr(retval, tree);
      tree = NULL; /* prevent freeing below */
      nsamp++;
      if (nsamp >= nsamples) keep_sampling = FALSE;
    }

    if (logf != NULL)
      fprintf(logf, "## %d\t%f\t%f\t%u\t%u\t%f\t%f\t%f\t%f\n", niters, lnl, lastlnl, burnin, accept,
              new_accrt, accrt, rho, s);

    oldtree = tree; /* set up for next iteration */
  }

  /* note that the last tree sampled must have been retained in
   * retval, so we don't have to worry about freeing it here; all
   * other trees have been freed during the loop */
  
  vec_free(mu);
  vec_free(lastz);
  vec_free(zprop);
  vec_free(zeta);
  vec_free(x);
  vec_free(y);
  
  return retval; 
}
