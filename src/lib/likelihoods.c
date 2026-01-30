/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/*
 * Portions of this file are derived from the PHAST package,
 * distributed under the BSD 3-Clause License.
 */

/* phylogenetic likelihood calculations for DNA models */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <nj.h>
#include <likelihoods.h>
#include <phast/sufficient_stats.h>

/* for multithreading */
#ifdef _OPENMP
  #include <omp.h>
  #define NJ_OMP_GET_MAX_THREADS() omp_get_max_threads()
  #define NJ_OMP_GET_THREAD_NUM() omp_get_thread_num()
#else
  #define NJ_OMP_GET_MAX_THREADS() 1
  #define NJ_OMP_GET_THREAD_NUM()  0
#endif

/* for use in likelihood calculations to avoid
	 underflow */
#define REL_CUTOFF 1e-300

/* fully reset a tree model for use in likelihood calculations with a
   new tree */
void nj_reset_tree_model(TreeModel *mod, TreeNode *newtree) {
  if (mod->tree != NULL)
    tr_free(mod->tree);

  mod->tree = newtree;
  
  /* the size of the tree won't change, so we don't need to do a full
     call to tm_reset_tree; we also don't need to call
     tm_set_subst_matrices because it will be called each time the
     likelihood function is invoked */
}

/* Compute and return the log likelihood of a tree model with respect
   to an alignment.  This is a pared-down version of
   tl_compute_log_likelihood from PHAST.  It assumes a 0th order
   model, sufficient stats already available, and no rate variation.
   It also makes use of scaling factors to avoid underflow with large
   trees.  If branchgrad is non-null, it will be populated with the
   gradient of the log likelihood with respect to the individual
   branches of the tree, in post-order.  This "core" routine is now
   threadsafe and is called by wrapper functions that handle setup and
   multithreading (see below).  The 'range' parameter can be used to
   specify a subset of tuple indices in a multithreading setting (set
   NULL to ignore). */
double nj_ll_core(TreeModel *mod, CovarData *data, Vector *branchgrad,
                  NJDerivs *derivs, NJGradCache *gcache, List *range) {

  int i, j, k, nodeidx, rcat = 0, tupleidx;
  int nstates = mod->rate_matrix->size;
  TreeNode *n;
  double total_prob = 0;
  List *traversal;
  double **pL = NULL, **pLbar = NULL;
  Vector *lscale, *lscale_o; /* inside and outside versions */
  double ll = 0;
  double tmp[nstates];
  MSA *msa = data->msa;
  Vector *this_deriv_gtr = NULL;
  Vector *tuplecounts = gcache->tuplecounts;
  
  pL = smalloc(nstates * sizeof(double*));
  for (j = 0; j < nstates; j++)
    pL[j] = smalloc((mod->tree->nnodes+1) * sizeof(double));

  /* we also need to keep track of the log scale of every node for
     underflow purposes */
  lscale = vec_new(mod->tree->nnodes+1); 
  lscale_o = vec_new(mod->tree->nnodes+1);
  
  if (branchgrad != NULL) {
    if (branchgrad->size != mod->tree->nnodes-1) /* rooted */
      die("ERROR in nj_compute_log_likelihood: size of branchgrad must be "
          "2n-2\n");
    vec_zero(branchgrad);
    pLbar = smalloc(nstates * sizeof(double*));
    for (j = 0; j < nstates; j++)
      pLbar[j] = smalloc((mod->tree->nnodes+1) * sizeof(double));
    if (mod->subst_mod == HKY85)
      derivs->deriv_hky_kappa = 0.0;
    else if (mod->subst_mod == REV) {
      this_deriv_gtr = vec_new(data->gtr_params->size);
      vec_zero(derivs->deriv_gtr);
    }
  }

  /* set up active range of tuples for this thread */
  int t0 = 0, t1 = msa->ss->ntuples;
  if (range != NULL) {
    if (lst_size(range) != 2) die("nj_ll_core: range must have size 2");
    t0 = lst_get_int(range, 0);
    t1 = lst_get_int(range, 1);
    if (t0 < 0) t0 = 0;
    if (t1 > msa->ss->ntuples) t1 = msa->ss->ntuples;
  }
  
  for (tupleidx = t0; tupleidx < t1; tupleidx++) {
    if (vec_get(tuplecounts, tupleidx) == 0) continue; /* important when subsampling */

    /* reset scale */
    vec_zero(lscale); vec_zero(lscale_o);
    
    traversal = tr_postorder(mod->tree);
    for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
      n = lst_get_ptr(traversal, nodeidx);

      if (n->lchild == NULL) {
        /* leaf: base case of recursion */
        int thisseq = mod->msa_seq_idx[n->id];
        
        if (thisseq < 0) 
          die("No match in alignment for leaf %s.\n", n->name);
        
        int state = mod->rate_matrix->
          inv_states[(int)ss_get_char_tuple(msa, tupleidx,
                                            thisseq, 0)];

        for (i = 0; i < nstates; i++) {
          if (state < 0 || i == state)
            pL[i][n->id] = 1;
          else
            pL[i][n->id] = 0;
        }
      }
      else {
        /* general recursive case */
        MarkovMatrix *lsubst_mat = mod->P[n->lchild->id][rcat];
        MarkovMatrix *rsubst_mat = mod->P[n->rchild->id][rcat];

	/* do this in a way that handles scaling.  first compute
           unnormalized inside values */
	for (i = 0; i < nstates; i++) {
	  double totl = 0.0, totr = 0.0;

	  for (j = 0; j < nstates; j++)
	    totl += pL[j][n->lchild->id] *
		    mm_get(lsubst_mat, i, j);

	  for (k = 0; k < nstates; k++)
	    totr += pL[k][n->rchild->id] *
		    mm_get(rsubst_mat, i, k);

	  pL[i][n->id] = totl * totr;
	}

	/* nodewise max-normalization across states */
	double maxv = 0.0;
	for (i = 0; i < nstates; i++)
	  if (pL[i][n->id] > maxv)
	    maxv = pL[i][n->id];

	/* propagate scaling from children */
	vec_set(lscale, n->id,
		vec_get(lscale, n->lchild->id) +
		vec_get(lscale, n->rchild->id));

	if (maxv > 0.0) {
	  /* normalize and update scale */
	  for (i = 0; i < nstates; i++)
	    pL[i][n->id] /= maxv;

	  vec_set(lscale, n->id,
		  vec_get(lscale, n->id) + log(maxv));
	}

        /* zero out tiny values to save time later */
        for (i = 0; i < nstates; i++)
	  if (pL[i][n->id] < REL_CUTOFF)
	    pL[i][n->id] = 0.0;        
      }
    }
  
    /* termination */
    total_prob = 0;
    for (i = 0; i < nstates; i++)
      total_prob += vec_get(mod->backgd_freqs, i) *
        pL[i][mod->tree->id] * mod->freqK[rcat];

    if (total_prob == 0.0)
      total_prob = REL_CUTOFF;  /* avoid log(0) */
    
    ll += (log(total_prob) + vec_get(lscale, mod->tree->id)) * vec_get(tuplecounts, tupleidx);

    if (!isfinite(ll))
      break;  /* can happen with zero-length branches;
                 make calling code deal with it */

    /* to compute gradients efficiently, need to make a second pass
       across the tree to compute "outside" probabilities */
    if (branchgrad != NULL) {
      double expon = 0;
      traversal = tr_preorder(mod->tree);

      for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
        n = lst_get_ptr(traversal, nodeidx);

        if (n->parent == NULL) { /* base case */
          double maxv = 0.0;
          for (i = 0; i < nstates; i++) {
            pLbar[i][n->id] = vec_get(mod->backgd_freqs, i);
            if (pLbar[i][n->id] > maxv)
	      maxv = pLbar[i][n->id];
	  }

          /* lscale_o[root] is already zero from vec_zero */
	  if (maxv > 0.0) {
	    for (i = 0; i < nstates; i++)
	      pLbar[i][n->id] /= maxv;
	    vec_set(lscale_o, n->id, log(maxv));
	  }
        }
        else {            /* recursive case */
          TreeNode *sibling = (n == n->parent->lchild ?
                               n->parent->rchild : n->parent->lchild);
          MarkovMatrix *par_subst_mat = mod->P[n->id][rcat];
          MarkovMatrix *sib_subst_mat = mod->P[sibling->id][rcat];

	  /* first form tmp[j] = sum_k pLbar(parent=j) * pL(sibling=k) * P_sib(j,k) */
	  for (j = 0; j < nstates; j++) {
	    tmp[j] = 0.0;
	    double a = pLbar[j][n->parent->id];

	    if (a == 0.0) continue;

	    for (k = 0; k < nstates; k++) {
	      double b = pL[k][sibling->id];
	      if (b > 0.0)
		tmp[j] += a * b * mm_get(sib_subst_mat, j, k);
	    }
	  }

	  /* now propagate to child */
	  for (i = 0; i < nstates; i++) {
	    double sum = 0.0;
	    for (j = 0; j < nstates; j++)
	      sum += tmp[j] * mm_get(par_subst_mat, j, i);
	    pLbar[i][n->id] = sum;
	  }

	  /* nodewise normalization of outside vector */
	  double maxv = 0.0;
	  for (i = 0; i < nstates; i++)
	    if (pLbar[i][n->id] > maxv)
	      maxv = pLbar[i][n->id];

	  /* bookkeeping for scaling */
	  vec_set(lscale_o, n->id,
		  vec_get(lscale_o, n->parent->id) +
		  vec_get(lscale, sibling->id));

	  if (maxv > 0.0) {
	    for (i = 0; i < nstates; i++)
	      pLbar[i][n->id] /= maxv;

	    vec_set(lscale_o, n->id,
		    vec_get(lscale_o, n->id) + log(maxv));
	  }

	  for (i = 0; i < nstates; i++)
	    if (pLbar[i][n->id] < REL_CUTOFF)
	      pLbar[i][n->id] = 0.0;
	}
      }

      /* TEMPORARY: check inside/outside */
      /* for (nodeidx = 0; nodeidx < lst_size(mod->tree->nodes); nodeidx++) { */
      /*   double pr = 0; */
      /*   n = lst_get_ptr(mod->tree->nodes, nodeidx); */
      /*   assert(vec_get(lscale, n->id) == 0 && vec_get(lscale_o, n->id) == 0); */
      /*   for (j = 0; j < nstates; j++) */
      /*     pr += pL[j][n->id] * pLbar[j][n->id]; */
      /*   printf("Tuple %d, node %d: %f (%f)\n", tupleidx, nodeidx, log(pr), log(total_prob)); */
      /* } */

      
      /* now compute branchwise derivatives in a final pass */
      traversal = mod->tree->nodes;
      for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
        TreeNode *par, *sib;
        double base_prob = total_prob, deriv;
        
        n = lst_get_ptr(traversal, nodeidx);
        par = n->parent;
	
        if (par == NULL) 
          continue;
       
        sib = (n == n->parent->lchild ?
               n->parent->rchild : n->parent->lchild);

        /* this part is just a constant to propagate through to the
           derivative */
        for (i = 0; i < nstates; i++) {  /* parent */
          tmp[i] = 0;
          for (k = 0; k < nstates; k++)  /* sibling */
            tmp[i] += pL[k][sib->id] * mm_get(mod->P[sib->id][rcat], i, k);
        }

        if (n != mod->tree->rchild) { /* skip branch to right of root because unrooted */
          /* calculate derivative analytically */
          deriv = 0;
          for (i = 0; i < nstates; i++)   
            for (j = 0; j < nstates; j++)    
              deriv +=  tmp[i] * pLbar[i][par->id] * pL[j][n->id] * mat_get(gcache->grad_mat[n->id], i, j);

          /* adjust for all relevant scale terms; do everything in log space */
          expon = -vec_get(lscale, mod->tree->id)
            + vec_get(lscale, sib->id) + vec_get(lscale_o, par->id)
            + vec_get(lscale, n->id) - log(base_prob);
          /* note division by base_prob because we need deriv of log P */

          /* avoid overflow */
          if (expon > 700.0) expon = 700.0;
          if (expon < -745.0) expon = -745.0;
          
          deriv *= exp(expon);
          assert(isfinite(deriv));
                  
          vec_set(branchgrad, n->id, vec_get(branchgrad, n->id) +
                  deriv * vec_get(tuplecounts, tupleidx));
        }

        /* in these cases, we need a partial derivatives for substitution rates also;
           they have to be aggregated across all branches */
        if (mod->subst_mod == HKY85) {
          double this_deriv_kappa = 0;
          for (i = 0; i < nstates; i++) 
            for (j = 0; j < nstates; j++) 
              this_deriv_kappa += tmp[i] * pLbar[i][par->id] * pL[j][n->id] *
                mat_get(gcache->grad_mat_HKY[n->id], i, j);

          /* adjust for all relevant scale terms */
          this_deriv_kappa *= exp(expon);
          derivs->deriv_hky_kappa +=
            (this_deriv_kappa * vec_get(tuplecounts, tupleidx));
        }
        else if (mod->subst_mod == REV) {
          /* loop over rate parameters */
          for (int pidx = 0; pidx < data->gtr_params->size; pidx++) {
            double pderiv = 0; /* partial deriv wrt this param */
            Matrix *dP_dr = lst_get_ptr(gcache->grad_mat_REV[n->id], pidx);
            for (int i = 0; i < nstates; i++) {
              for (int j = 0; j < nstates; j++) {
                pderiv += tmp[i] * pLbar[i][par->id] * pL[j][n->id] *
                  mat_get(dP_dr, i, j);
              }
            }
            vec_set(this_deriv_gtr, pidx, pderiv);
          }
          /* adjust for all relevant scale terms */
          vec_scale(this_deriv_gtr, exp(expon));
          vec_plus_eq(derivs->deriv_gtr, this_deriv_gtr);
        }
      }
    }
  }
  
  for (j = 0; j < nstates; j++)
    sfree(pL[j]);
  sfree(pL);
  
  if (branchgrad != NULL) {
    for (j = 0; j < nstates; j++)
      sfree(pLbar[j]);
    sfree(pLbar);
    if (mod->subst_mod == REV)
      vec_free(this_deriv_gtr);
  }

  vec_free(lscale);
  vec_free(lscale_o);
  
  return ll;
}

/* Build index of leaf ids to sequence indices based on a name
   list. */
int *nj_build_seq_idx(List *leaves, char **names) {
  int i;  
  int *retval = smalloc(lst_size(leaves)*2 * sizeof(int));
  for (i = 0; i < lst_size(leaves)*2; i++) retval[i] = -1;
  for (i = 0; i < lst_size(leaves); i++) {
    TreeNode *n = lst_get_ptr(leaves, i);
    retval[n->id] = nj_get_seq_idx(names, n->name, lst_size(leaves));
    if (retval[n->id] < 0)
      die("ERROR: leaf '%s' not found in name list.\n", n->name);
  }
  return retval;
}

/* Return index of given sequence name or -1 if not found. */
int nj_get_seq_idx(char **names, char *name, int n) {
  int i, retval = -1;
  for (i = 0; retval < 0 && i < n; i++) 
    if (!strcmp(name, names[i]))
      retval = i;
  return retval;
}

/* custom setup for parameter mapping for GTR parameterization.  Set
   up lists that allow each rate matrix parameter to be mapped to the
   rows and columns in which it appears; this is a simplified version
   of tm_init_rmp */
void nj_init_gtr_mapping(TreeModel *tm) {
  tm->rate_matrix_param_row = (List**)smalloc(GTR_NPARAMS * sizeof(List*));
  tm->rate_matrix_param_col = (List**)smalloc(GTR_NPARAMS * sizeof(List*));
  for (int i = 0; i < GTR_NPARAMS; i++) {
    tm->rate_matrix_param_row[i] = lst_new_int(2);
    tm->rate_matrix_param_col[i] = lst_new_int(2);
  }
}

/* --- code for multithreading of likelihood calculations --- */

/* The main likelihood function (same interface as previously) is now a wrapper
   that handles setup, subsampling, and multithreading.  The actual likelihood
   calculation is done in nj_ll_core, which is called by
   each thread separately. */
double nj_compute_log_likelihood(TreeModel *mod, CovarData *data,
                                 Vector *branchgrad) {
  double ll;

  /* ---- basic sanity checks  ---- */
 if (data->msa->ss->tuple_size != 1)
    die("ERROR nj_compute_log_likelihood: need tuple size 1, got %i\n",
	data->msa->ss->tuple_size);
  if (mod->order != 0)
    die("ERROR nj_compute_log_likelihood: got mod->order of %i, expected 0\n",
	mod->order);
  if (!mod->allow_gaps)
    die("ERROR nj_compute_log_likelihood: need mod->allow_gaps to be TRUE\n");
  if (mod->nratecats > 1)
    die("ERROR nj_compute_log_likelihood: no rate variation allowed\n");
    
  /* ---- one-time model setup (not thread-safe if repeated) ---- */
  tm_set_subst_matrices(mod);

  if (mod->msa_seq_idx == NULL)
    tm_build_seq_idx(mod, data->msa);

  tr_postorder(mod->tree); /* ensure these are cached */
  tr_preorder(mod->tree);
  
  /* ---- policy enforcement ---- */
  if (data->nthreads > 1 && data->subsample)
    die("ERROR: subsampling is not allowed with multithreading.\n");

  /* set up tuplecounts; subsample if needed */
  static Vector *tuplecdf = NULL;
  static Vector *tuplecounts = NULL;
  if (data->subsample == TRUE) {
    if (tuplecdf == NULL) { /* build CDF for sampling */
      Vector *counts = vec_view_array(data->msa->ss->counts, data->msa->ss->ntuples);
      tuplecdf = pv_cdf_from_counts(counts, LOWER);
      sfree(counts); /* avoid vec_free */
    }
    else  /* CDF already exists; check that it makes sense */
      assert(tuplecdf->size == data->msa->ss->ntuples);
    if (tuplecounts == NULL || data->reuse_subsamp == FALSE ||
	vec_sum(tuplecounts) != data->subsampsize) { /* need new subsample */
      if (tuplecounts == NULL) tuplecounts = vec_new(data->msa->ss->ntuples);
      else assert(tuplecounts->size == data->msa->ss->ntuples);
      pv_draw_counts(tuplecounts, tuplecdf, data->subsampsize);
    }
  }
  else 
    /* not subsampling; just provide a 'view' of the full counts array */
    tuplecounts = vec_view_array(data->msa->ss->counts, data->msa->ss->ntuples);

  /* ---- set up gradient cache ---- */
  int nnodes = mod->tree->nnodes;
  int nstates = mod->rate_matrix->size;
  NJGradCache gcache = {0};
  gcache.tuplecounts = tuplecounts;
  if (branchgrad != NULL) {
    int j, p;
    gcache.grad_mat = malloc(nnodes * sizeof(Matrix *));
    for (j = 0; j < nnodes; j++)
      gcache.grad_mat[j] = mat_new(nstates, nstates);

    if (mod->subst_mod == HKY85) {
      gcache.grad_mat_HKY = malloc(nnodes * sizeof(Matrix *));
      for (j = 0; j < nnodes; j++)
	gcache.grad_mat_HKY[j] = mat_new(nstates, nstates);
    }
    else if (mod->subst_mod == REV) {
      gcache.grad_mat_REV = malloc(nnodes * sizeof(List *));
      for (j = 0; j < nnodes; j++) {
	gcache.grad_mat_REV[j] = lst_new_ptr(data->gtr_params->size);
	for (p = 0; p < data->gtr_params->size; p++)
          lst_push_ptr(gcache.grad_mat_REV[j], mat_new(nstates, nstates));
      }
    }
    /* compute gradient matrices ONCE */
    for (j = 0; j < nnodes; j++) {
      TreeNode *n = lst_get_ptr(mod->tree->nodes, j);

      if (mod->subst_mod == JC69)
        tm_grad_JC69(mod, gcache.grad_mat[j], n->dparent);
      else if (mod->subst_mod == HKY85) {
        tm_grad_HKY_dt(mod, gcache.grad_mat[j], data->hky_kappa, n->dparent);
        tm_grad_HKY_dkappa(mod, gcache.grad_mat_HKY[j], data->hky_kappa,
                           n->dparent);
      } else if (mod->subst_mod == REV) {
        tm_grad_REV_dt(mod, gcache.grad_mat[j], n->dparent);
        tm_grad_REV_dr(mod, gcache.grad_mat_REV[j], n->dparent);
      }
    }
  }

  /* if multiple threads are requested but OpenMP is not active, catch it here */
#ifndef _OPENMP
  if (data->nthreads > 1) 
    die("ERROR: Multithreading requested but OpenMP is not enabled.\n");
#endif
  
  /* ---- sequential path ---- */
  if (data->nthreads == 1) 
    /* legacy behavior, but wrapper-controlled */
    ll = nj_ll_parallel(mod, data, branchgrad, 1, &gcache);

  /* ---- parallel path ---- */
  else
    ll = nj_ll_parallel(mod, data, branchgrad, data->nthreads, &gcache);

  if (data->subsample == FALSE) {
    sfree(tuplecounts);  /* view wrapper */
    tuplecounts = NULL;
  }
    
  if (branchgrad != NULL) {
    for (int j = 0; j < nnodes; j++)
      mat_free(gcache.grad_mat[j]);
    free(gcache.grad_mat);

    if (gcache.grad_mat_HKY) {
      for (int j = 0; j < nnodes; j++)
        mat_free(gcache.grad_mat_HKY[j]);
      free(gcache.grad_mat_HKY);
    }

    if (gcache.grad_mat_REV) {
      for (int j = 0; j < nnodes; j++) {
        List *gm = gcache.grad_mat_REV[j];
        for (int p = 0; p < lst_size(gm); p++)
          mat_free(lst_get_ptr(gm, p));
        lst_free(gm);
      }
      free(gcache.grad_mat_REV);
    }
  }

  return ll;
}

double nj_ll_parallel(TreeModel *mod, CovarData *data, Vector *branchgrad,
                      int nthreads_requested, NJGradCache *gcache) {

  int ntuples = data->msa->ss->ntuples;

  int maxthreads = NJ_OMP_GET_MAX_THREADS();
  int nthreads = (nthreads_requested <= maxthreads ? nthreads_requested : maxthreads);

  double ll_total = 0.0;

  /* ---- allocate per-thread accumulators ---- */
  Vector **thread_branchgrad = NULL;
  NJDerivs **thread_derivs = NULL;
  double *thread_ll = calloc(nthreads, sizeof(double));

  if (branchgrad != NULL) {
    thread_branchgrad = malloc(nthreads * sizeof(Vector *));
    thread_derivs = malloc(nthreads * sizeof(NJDerivs *));

    for (int t = 0; t < nthreads; t++) {
      thread_branchgrad[t] = vec_new(branchgrad->size);
      vec_zero(thread_branchgrad[t]);

      thread_derivs[t] = malloc(sizeof(NJDerivs));
      thread_derivs[t]->deriv_hky_kappa = 0.0;
      thread_derivs[t]->deriv_gtr =
          (mod->subst_mod == REV ? vec_new(data->gtr_params->size) : NULL);
    }
  }

  /* ---- parallel likelihood computation ---- */
#pragma omp parallel num_threads(nthreads)
  {
    int tid = NJ_OMP_GET_THREAD_NUM();

    int start = (tid * ntuples) / nthreads;
    int end = ((tid + 1) * ntuples) / nthreads;

    List *range = lst_new_int(2);
    lst_push_int(range, start);
    lst_push_int(range, end);

    thread_ll[tid] =
        nj_ll_core(mod, data, (branchgrad ? thread_branchgrad[tid] : NULL),
                   (branchgrad ? thread_derivs[tid] : NULL), gcache, range);

    lst_free(range);
  }

  /* ---- reduction ---- */
  for (int t = 0; t < nthreads; t++)
    ll_total += thread_ll[t];

  if (branchgrad != NULL) {
    vec_zero(branchgrad);
    for (int t = 0; t < nthreads; t++)
      vec_plus_eq(branchgrad, thread_branchgrad[t]);

    data->deriv_hky_kappa = 0.0;
    for (int t = 0; t < nthreads; t++)
      data->deriv_hky_kappa += thread_derivs[t]->deriv_hky_kappa;

    if (mod->subst_mod == REV) {
      vec_zero(data->deriv_gtr);
      for (int t = 0; t < nthreads; t++)
        vec_plus_eq(data->deriv_gtr, thread_derivs[t]->deriv_gtr);
    }
  }

  /* ---- cleanup ---- */
  free(thread_ll);

  if (branchgrad != NULL) {
    for (int t = 0; t < nthreads; t++) {
      vec_free(thread_branchgrad[t]);
      if (thread_derivs[t]->deriv_gtr)
        vec_free(thread_derivs[t]->deriv_gtr);
      free(thread_derivs[t]);
    }
    free(thread_branchgrad);
    free(thread_derivs);
  }

  return ll_total;
}
