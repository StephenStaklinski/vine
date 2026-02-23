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

#define BURNIN_ITERS 200
#define TUNING_INTERVAL 10
#define TARGET_ACCEPT_RATE 0.3

/* MCMC refinement of variational samples */

List *nj_var_sample_mcmc(int nsamples, int thin, multi_MVN *mmvn,
                         CovarData *data, TreeModel *mod, FILE *logf) {

  int n = data->nseqs, dim = data->dim, fulld = n * dim;
  int niters = 0, naccept = 0, last_naccept = 0, nsamp = 0;
  unsigned int keep_sampling = TRUE, burnin = TRUE;
  double rho = 0.99, s = 1.0, accrt = 0.0; 
  Vector *mu = vec_new(fulld);
  double nf_logdet;
  mmvn_save_mu(mmvn, mu);
  TreeNode *tree;
  List *retval = lst_new_ptr(nsamples);

  if (n >= 250) rho = 0.995; /* more conservative for higher dimensions */
  
  /* what to log? acceptance rate, log l, whether or not new sample.  Burnin. s and rho.  Put after #*/

  /* FIXME: work on lower dimension in LOWR case? I think affects z */
  /* CHECK: does this extend to hyperbolic? */
  
  /* initialize last_lnl based on mean; can we do this below by being clever? */
  double lnl = 0.0, lastlnl = 0.0; /* placeholder */

  /* how to handle burn-in? */
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
    vec_plus_eq(x, mu);  /* CHECK: does this work in LOWR case? */    
    
    /* convert x to y using normalizing flows if available */
    nj_apply_normalizing_flows(y, x, data, &nf_logdet);

    /* FIXME: check that scaling is correct */
    
    /* set up baseline objects */
    nj_points_to_distances(y, data);
    tree = nj_inf(data->dist, data->names, NULL, NULL, data);
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
    
    double alpha = fmin(1.0, exp(lnl - lastlnl));
    if (unif_rand() < alpha) {
      /* accept */
      lastlnl = lnl;
      vec_copy(lastz, zprop);
      naccept++;
    }
    accrt = naccept / (double)niters;

    /* during burnin only, adapt s and rho to target acceptance rate;
     * after burnin, keep fixed */
    if (burnin && niters % TUNING_INTERVAL == 0) {
      int new_naccept =
          naccept - last_naccept; /* number accepted since last check */
      double new_accrt =
        new_naccept / (double)TUNING_INTERVAL; /* acceptance rate since last check */
      int t = niters / (double)TUNING_INTERVAL; /* number of checks so far */

      /* diminishing learning rates */
      double eta_rho = 0.10 / sqrt((double)t);   
      double eta_s   = 0.03 / sqrt((double)t);

      /* update rho on log scale (keeps rho>0) */
      rho = exp(log(rho) + eta_rho * (new_accrt - TARGET_ACCEPT_RATE));
      if (rho < 0.90)   rho = 0.90;
      if (rho > 0.9995) rho = 0.9995;

      /* update s gently, and only if rho is near a bound */
      if ((rho > 0.999 || rho < 0.905)) {
        s = exp(log(s) + eta_s * (new_accrt - TARGET_ACCEPT_RATE));
        if (s < 0.1) s = 0.1;
        if (s > 10)  s = 10;
      }

      last_naccept = naccept;
    }

    if (!burnin && niters % thin == 0) {
      /* FIXME: stash sample */
      nsamp++;
      if (nsamp == nsamples) keep_sampling = FALSE;
    }
    
    /* FIXME: add trees to retval every thin iterations; figure out
     * memory management of trees */
    /* I think reset_tree_model calls tr_free; but call explicitly for others */
    /* we have to store the prev one and keep a copy if we reject the proposal
     */
    /* check previous sampling routines for how to do this */
  }

  vec_free(mu);
  vec_free(lastz);
  vec_free(zprop);
  vec_free(zeta);
  vec_free(x);
  vec_free(y);
  
  return retval; 
}
