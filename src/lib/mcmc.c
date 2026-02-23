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
#include <geometry.h>
#include <variational.h>
#include <mvn.h>
#include <mcmc.h>
#include <multi_mvn.h>

List *nj_var_sample_mcmc(int nsamples, int thin, multi_MVN *mmvn,
                         CovarData *data, TreeModel *mod, FILE *logf) {
  return NULL; /* placeholder */
}
