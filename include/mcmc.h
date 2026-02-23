/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* MCMC refinement of variational samples */

#ifndef MCMC_H
#define MCMC_H

#include <stdio.h>
#include <phast/tree_model.h>
#include <multi_mvn.h>
#include <nj.h>

List *nj_var_sample_mcmc(int nsamples, int thin, multi_MVN *mmvn,
                         CovarData *data, TreeModel *mod, FILE *logf);

#endif
