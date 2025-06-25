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
Script containing the main functions for the regression and control applications.
"""

import multiprocess
import numpy as np
from src.pop_classes import POP_geno
from src.utils import HallOfFame_modified, Min
from src.evolutionary_strategies import eaMuPlusLambdaTol
from deap import tools


def main_IGP_regression(size_pop, size_gen, Mu, Lambda, cxpb, mutpb, nbCPU, X_train, y_train, X_val, y_val,pset,
                        creator, toolbox, save_path_iter, save_pop, save_gen, **kwargs):

    fit_tol = kwargs['kwargs']['fit_tol']
    terminals = kwargs['kwargs']['terminals']
    cx_lim = kwargs['kwargs']['cx_lim']

    if nbCPU == 1:
        toolbox.register('map', map)
    else:
        pool = multiprocess.Pool(nbCPU)
        toolbox.register('map', pool.map)
    old_entropy = 0
    for i in range(200):
        pop = POP_geno(toolbox.population(n=size_pop), creator)
        best_pop = pop.items
        if pop.entropy_len > old_entropy and len(pop.indexes_len) == len(pop.categories_len) - 1:
            best_pop = pop.items
            old_entropy = pop.entropy_len

    hof = HallOfFame_modified(10)

    print("INITIAL POP SIZE: %d" % size_pop)
    print("GEN SIZE: %d" % size_gen)
    print("\n")

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_fit_val = tools.Statistics(lambda ind: ind.fitness_validation.values)
    mstats = tools.MultiStatistics(fitness=stats_fit, fitness_val=stats_fit_val)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", Min)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   ###################################

    pop, log, pop_statistics, ind_lengths, hof = eaMuPlusLambdaTol(best_pop, toolbox, Mu, Lambda, size_gen, cxpb, mutpb,
                                                                   pset, creator, stats=mstats, X_train=X_train,
                                                                   y_train=y_train, X_val=X_val, y_val=y_val,
                                                                   save_gen=save_gen, fit_tol=fit_tol,
                                                                   terminals=terminals, halloffame=hof, cx_lim=cx_lim,
                                                                   verbose=True, save_path=save_path_iter,
                                                                   save_pop=save_pop)

    ####################################################################################################################
    if nbCPU != 1:
        pool.close()
        pool.join()
    return pop, log, hof, pop_statistics, ind_lengths, pset

