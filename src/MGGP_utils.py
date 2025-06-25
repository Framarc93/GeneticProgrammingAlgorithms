# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2025 Francesco Marchetti -----------------------
# ---------------- Author: Francesco Marchetti ------------------------
# ---------------- e-mail: framarc93@gmail.com ------------------------

# Alternatively, the contents of this file may be used under the terms
# of the GNU General Public License Version 3.0, as described below:

# This file is free software: you may copy, redistribute and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3.0 of the License, or (at your
# option) any later version.

# This file is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

"""
File containing the functions used by the MGGP algorithm

"""

import numpy as np
from deap import gp


def lst_matrix(ind, evaluate_subtree, evaluate, compile, input_train, output_train, input_val, output_val):
    Y = np.reshape(output_train, (len(output_train), 1))
    A = np.ones((len(output_train), len(ind) + 1))
    for i in range(len(ind)):
        A[:, i+1] = evaluate_subtree(ind, i, compile, input_train, output_train)
    try:
        c = np.linalg.pinv(A) @ Y
        if np.isnan(c).any() or np.isinf(c).any():
            ind.fitness.values = 1e6,
            ind.fitness_validation.values = 1e6,
            return ind
        c = np.ndarray.flatten(c)
        ind.w = c
        err_train, err_val = evaluate(c, ind, compile, input_train, output_train, input_val, output_val)
        fit_train = np.sqrt(np.sum(err_train ** 2)/ len(err_train))  # output-output_eval
        fit_val = np.sqrt(np.sum(err_val ** 2) / len(err_val))  # output-output_eval
        ind.fitness.values = fit_train,
        ind.fitness_validation.values = fit_val,
    except np.linalg.LinAlgError:
        ind.fitness.values = 1e6,
        ind.fitness_validation.values = 1e6,
    return ind


def evaluate_subtree(individual, gene, compile, input_true, output_true):
    f_ind = compile(individual[gene])
    out = f_ind(*input_true)
    if not hasattr(out, "__len__"):
        out = np.ones(len(output_true)) * out
    return out


def build_funcString(d, individual):
    ws = 1
    eq = str(d[0]) + "+"
    while ws < len(d):
        eq = eq + str(d[ws]) + "*" + str(individual[ws - 1]) + "+"
        ws += 1
    eq = list(eq)
    del eq[-1]
    eq = "".join(eq)
    return eq



