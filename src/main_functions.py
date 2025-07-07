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
from deap import tools
import numpy as np
from src.utils import HallOfFame_modified, Min


def main_regression(size_pop, size_gen, Mu, Lambda, cxpb, mutpb, nbCPU, X_train, y_train, X_val, y_val,pset,
                        creator, toolbox, save_path_iter, **kwargs):

    kwargs = kwargs['kwargs']
    kwargs['X_train'] = X_train
    kwargs['y_train'] = y_train
    kwargs['X_val'] = X_val
    kwargs['y_val'] = y_val
    kwargs['save_path'] = save_path_iter

    init_repeat = kwargs['init_repeat']

    if nbCPU == 1:
        toolbox.register('map', map)
    else:
        pool = multiprocess.Pool(nbCPU)
        toolbox.register('map', pool.map)

    if hasattr(toolbox, "pop_init"):
        best_pop = toolbox.pop_init(size_pop, toolbox, creator, init_repeat, kwargs=kwargs)
    else:
        best_pop = toolbox.population(size_pop)

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

    pop, log, pop_statistics, ind_lengths, hof = toolbox.evol_strategy(best_pop, toolbox, Mu, Lambda, size_gen, cxpb, mutpb,
                                                                   pset, creator, stats=mstats, halloffame=hof,
                                                                       verbose=True, kwargs=kwargs)

    ####################################################################################################################
    if nbCPU != 1:
        pool.close()
        pool.join()
    return pop, log, hof, pop_statistics, ind_lengths, pset

