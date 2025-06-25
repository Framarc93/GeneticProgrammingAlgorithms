# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------ Copyright (C) 2020 University of Strathclyde and Author ------
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

import multiprocess
from functools import partial
from deap import gp
from copy import copy
from src.pop_classes import POP_pheno_3D_2fit
import numpy as np
from src.utils import HallOfFame_modified, Min
from deap import tools
from src.evolutionary_strategies import eaMuPlusLambdaTol_pheno_2fit

def main_FIGP_regression(size_pop, size_gen, Mu, Lambda, cxpb, mutpb, nbCPU, X_train, y_train, X_val, y_val,pset,
                        creator, toolbox, save_path_iter, save_pop, save_gen, **kwargs):

    cat_number_len = kwargs['kwargs']['cat_number_len']
    cat_number_fit = kwargs['kwargs']['cat_number_fit']
    cat_number_height = kwargs['kwargs']['cat_number_height']
    fit_scale = kwargs['kwargs']['fit_scale']
    terminals = kwargs['kwargs']['terminals']
    fit_tol = kwargs['kwargs']['fit_tol']
    cx_lim = kwargs['kwargs']['cx_lim']

    if nbCPU == 1:
        toolbox.register('map', map)
    else:
        pool = multiprocess.Pool(nbCPU)
        toolbox.register("map", pool.map)

    best_pop = []

    old_entropy = 0

    for i in range(100):
        pop = toolbox.population(n=size_pop)

        fitnesses = toolbox.map(partial(toolbox.evaluate, pset=pset, compile=gp.compile, kwargs={'X_train':X_train,
                                        'y_train':y_train, 'X_val':X_val, 'y_val':y_val}), pop)

        # assign evaluated fitness to population

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit[:2]
            ind.fitness_validation.values = fit[2:]

        pop_pheno = copy(POP_pheno_3D_2fit(pop, creator, cat_number_len, cat_number_fit, cat_number_height, fit_scale))
        best_pop = pop_pheno.items
        entropy = pop_pheno.entropy

        if entropy > old_entropy:
            best_pop = copy(pop_pheno.items)
            old_entropy = copy(entropy)

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

    pop, log, pop_statistics, ind_lengths = eaMuPlusLambdaTol_pheno_2fit(best_pop, toolbox, Mu, Lambda, size_gen, cxpb,
                                                                         mutpb, pset, creator, stats=mstats,
                                                                         halloffame=hof, verbose=True, X_train=X_train,
                                                                         y_train=y_train, X_val=X_val, y_val=y_val,
                                                                         terminals=terminals, save_gen=save_gen,
                                                                         save_path=save_path_iter, fit_tol=fit_tol,
                                                                         cx_lim=cx_lim, cat_number_len=cat_number_len,
                                                                         cat_number_fit=cat_number_fit,
                                                                         cat_number_height=cat_number_height,
                                                                         fit_scale=fit_scale, save_pop=save_pop)
    ####################################################################################################################

    if nbCPU != 1:
        pool.close()
        pool.join()

    return pop, log, hof, pop_statistics, ind_lengths, pset