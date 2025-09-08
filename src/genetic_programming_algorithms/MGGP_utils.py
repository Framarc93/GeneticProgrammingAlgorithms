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


def lst_matrix(ind, evaluate_subtree, evaluate, compile, **kwargs):

    input_train = kwargs['kwargs']['X_train']
    output_train = kwargs['kwargs']['y_train']

    Y = np.reshape(output_train, (len(output_train), 1))
    A = np.ones((len(output_train), len(ind) + 1))
    for i in range(len(ind)):
        A[:, i+1] = evaluate_subtree(ind, i, compile, input_train, output_train)
    try:
        c = np.linalg.pinv(A) @ Y
        if np.isnan(c).any() or np.isinf(c).any():
            fitness_train = 1e6
            fitness_val = 1e6
            c = np.zeros(len(ind) + 1)
            return fitness_train, 0.0, fitness_val, 0.0, c
        c = np.ndarray.flatten(c)
        ind.w = c
        eq = build_funcString(c, ind)
        fitnesses = evaluate(eq, compile, kwargs=kwargs['kwargs'])
        fitness_train = fitnesses[:2]
        fitness_val = fitnesses[2:]
    except np.linalg.LinAlgError:
        fitness_train = 1e6
        fitness_val = 1e6
        c = np.zeros(len(ind)+1)
    return fitness_train, 0.0, fitness_val, 0.0, c


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



