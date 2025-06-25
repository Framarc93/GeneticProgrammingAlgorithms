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
Script containing the main functions for the regression and control applications of the MGGP algorithm.
"""

import multiprocess
import numpy as np
from functools import partial
from src.MGGP_utils import lst_matrix, evaluate_subtree
from src.recombination_functions import varOr
from copy import deepcopy
from src.pop_classes import POP_geno
from deap.tools import selBest
from examples.regression.evaluate_functions import evaluate_MGGP
import dill
from deap import tools
from src.utils import Min

def main_MGGP_regression(size_pop, size_gen, Mu, Lambda, cxpb, mutpb, nbCPU, X_train, y_train, X_val, y_val, pset,
                         creator, toolbox, save_path_iter, save_pop,  save_gen, **kwargs):
                         #terminals,
                         # fit_tol, cx_lim, cat_number_fit, cat_number_height, cat_number_len, fit_scale,
                         # NgenesMax, stdCxpb):



    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_fit_val = tools.Statistics(lambda ind: ind.fitness_validation.values)
    mstats = tools.MultiStatistics(fitness=stats_fit, fitness_val=stats_fit_val)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", Min)

    if nbCPU == 1:
        toolbox.register('map', map)
    else:
        pool = multiprocess.Pool(nbCPU)
        toolbox.register("map", pool.map)

    data = np.array(['Min length', 'Max length', 'Entropy', 'Distribution'])
    all_lengths = []
    min_fits = []

    pop = toolbox.population(size_pop)  # creation of initial population
    pop = list(toolbox.map(partial(lst_matrix, evaluate_subtree=evaluate_subtree, output_train=y_train, evaluate=evaluate_MGGP,
                              compile=toolbox.compile, input_train=X_train, input_val=X_val, output_val=y_val), pop))

    pop_stat = POP_geno(pop, creator)
    data, all_lengths = pop_stat.retrieve_stats(data, all_lengths)

    gen = 0

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])

    record = mstats.compile(pop) if mstats is not None else {}
    logbook.record(gen=gen, nevals=len(pop), **record)
    print(logbook.stream)

    best_ind = selBest(pop, 1)[0]
    best_fit = best_ind.fitness.values[0]
    best_ind_allTime = deepcopy(best_ind)
    best_fit_allTime = deepcopy(best_fit)
    min_fits.append(best_fit_allTime)

    # save data
    if (save_gen is not None) and (save_gen == True):
        output = open(save_path_iter + 'Best_ind_{}'.format(gen), "wb")
        dill.dump(best_ind, output, -1)
        output.close()

    if (save_pop is not None) and (save_pop == True):
        output = open(save_path_iter + 'Full_population_{}'.format(gen), "wb")
        dill.dump(pop, output, -1)
        output.close()

    gen+=1

    while gen < size_gen and best_fit_allTime > 0:

        #print("------------------------------------------------------------------------------------------------------------- GEN {}".format(gen))

        offspring = varOr(pop, toolbox, Lambda, cxpb, mutpb)

        offspring = list(toolbox.map(partial(lst_matrix, evaluate_subtree=evaluate_subtree, output_train=y_train,
                                        evaluate=evaluate_MGGP, compile=toolbox.compile, input_train=X_train,
                                        input_val=X_val, output_val=y_val), offspring))

        global_pop = pop + offspring
        best_ind = selBest(global_pop, 1)
        pop_without_best = [ind for ind in global_pop if ind != best_ind]

        # compute statistics on population
        pop[:] = toolbox.select(pop_without_best, Mu - 1)
        pop = pop + best_ind

        pop_stat = POP_geno(pop, creator)
        data, all_lengths = pop_stat.retrieve_stats(data, all_lengths)

        best_fit = best_ind[0].fitness.values[0]
        if best_fit < best_fit_allTime:
            best_fit_allTime = deepcopy(best_fit)
            best_ind_allTime = deepcopy(best_ind[0])
        min_fits.append(best_fit_allTime)
        #print("------------------------------------------------------------------------------------------------------------- {}".format(best_fit_allTime))
        #string = str(best_ind_allTime.w[0])
        #st = 1
        #while st <= len(best_ind_allTime):
        #    string = string + "+" + str(best_ind_allTime.w[st]) + "*" + str(best_ind_allTime[st-1])
        #    st += 1
        #print(string)

        # Update the statistics with the new population
        record = mstats.compile(pop) if mstats is not None else {}
        logbook.record(gen=gen, nevals=len(pop), **record)
        print(logbook.stream)

        # save data
        if (save_gen is not None) and (save_gen == True):
            output = open(save_path_iter + 'Best_ind_{}'.format(gen), "wb")
            dill.dump(best_ind[0], output, -1)
            output.close()

        if (save_pop is not None) and (save_pop == True):
            output = open(save_path_iter + 'Full_population_{}'.format(gen), "wb")
            dill.dump(pop, output, -1)
            output.close()

        gen += 1

    if nbCPU != 1:
        pool.close()
        pool.join()
    return pop, logbook, [0, best_ind_allTime], data, all_lengths, pset

