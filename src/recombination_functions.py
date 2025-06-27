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

from src.niches_manipulation import subset_feasible
from copy import copy, deepcopy
import random
import numpy as np
from src.selection_functions import selBest_IGP


def varOr_IGP(population, toolbox, lambda_, sub_div, good_indexes_original, cxpb, mutpb, verbose, cx_lim):
    """
    Function to perform crossover,mutation or reproduction operations. This is a modified version of the varOr function from DEAP
    library (https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L192C1-L192C54).

    Modified to implement the inclusive crossover, mutation and reproduction.

    Parameters:
        population : list
            A list of individuals to vary.
        toolbox : class, deap.base.Toolbox
            contains the evolution operators.
        lambda_ : int
            The number of children to produce
        sub_div : dict
            categories in which the population is divided
        good_indexes_original : int
            array contating the indexes of the filled categories
        cxpb : float
            The probability of mating two individuals.
        mutpb : float
            The probability of mutating an individual.
        limit_size : int
            size limit used to accept or not the mutation performed on an individual
    """
    assert (cxpb + mutpb) == 1.0, (
        "The sum of the crossover and mutation probabilities must be equal to 1.0. The best individual is always passsed")

    offspring = []

    # retrieve subset of feasible individuals. Feasible means that the second fitness, i.e. the penalty, is 0.
    sub_pop = subset_feasible(population)

    len_subpop = len(sub_pop)

    # if no feasible individuals are found, keep the mutation rate high to explore the search space
    if sub_pop == []:
        if verbose is True:
            print("Exploring for feasible individuals. Mutpb: {}, Cxpb:{}".format(mutpb, cxpb))
    else:
        # if feasible individuals are found, start decreasing the mutation rate and increasing the crossover rate, to improve exploitation
        if cxpb < cx_lim:
            mutpb = mutpb - 0.01
            cxpb = cxpb + 0.01

        if verbose is True:
            print("{}/{} ({}%) FEASIBLE INDIVIDUALS".format(len_subpop, len(population),
                                                            round(len(sub_pop) / len(population) * 100, 2)))
            print("Mutpb: {}, Cxpb:{}".format(mutpb, cxpb))
            print("\n")

    # assign indexes of filled niches
    good_indexes = copy(list(good_indexes_original))
    good_list = copy(list(good_indexes))

    while len(offspring) < lambda_:
        op_choice = random.random()
        if op_choice < cxpb:  # Apply crossover
            # selection of 2 different niches for crossover
            cat = np.zeros((2))
            for i in range(2):
                if not good_list:
                    good_list = list(good_indexes)
                used = random.choice(good_list)
                cat[i] = used
                good_list.remove(used)
            # select best individual from first niche and random one from the second
            ind1, ind2 = map(toolbox.clone, [selBest_IGP(sub_div["cat{}".format(int(cat[0]))]),
                                             random.choice(sub_div["cat{}".format(int(cat[1]))])])
            tries = 0
            while sum(ind1.fitness.values) == sum(
                    ind2.fitness.values) and tries < 10:  # if the same individual is selected, then repeat the niches selection
                # process for a maximum of 10 times
                # selection of 2 different niches for crossover
                cat = np.zeros((2))
                for i in range(2):
                    if not good_list:
                        good_list = list(good_indexes)
                    used = random.choice(good_list)
                    cat[i] = used
                    good_list.remove(used)
                # select best individual from first niche and random one from the second
                ind1, ind2 = map(toolbox.clone, [selBest_IGP(sub_div["cat{}".format(int(cat[0]))]),
                                                 random.choice(sub_div["cat{}".format(int(cat[1]))])])
                tries += 1
            # perform crossover on selected individuals
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
            if len(offspring) < lambda_:
                del ind2.fitness.values
                offspring.append(ind2)
        else:  # Apply mutation
            if not good_list:
                good_list = list(good_indexes)
            used = random.choice(good_list)
            cat = used
            good_list.remove(used)
            ind = toolbox.clone(random.choice(sub_div["cat{}".format(int(cat))]))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)

    return offspring, len_subpop, mutpb, cxpb


def varOr_FIGP(population, toolbox, lambda_, sub_div, good_indexes_original, cxpb, mutpb, verbose, cx_lim):
    """
    Function to perform crossover,mutation or reproduction operations. This is a modified version of the varOr function from DEAP
    library (https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L192C1-L192C54).

    Modified to implement the inclusive crossover, mutation and reproduction.

    Parameters:
        population : list
            A list of individuals to vary.
        toolbox : class, deap.base.Toolbox
            contains the evolution operators.
        lambda_ : int
            The number of children to produce
        sub_div : dict
            categories in which the population is divided
        good_indexes_original : int
            array contating the indexes of the filled categories
        cxpb : float
            The probability of mating two individuals.
        mutpb : float
            The probability of mutating an individual.
        limit_size : int
            size limit used to accept or not the mutation performed on an individual
    """
    assert (cxpb + mutpb) == 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []

    # retrieve subset of feasible individuals. Feasible means that the second fitness, i.e. the penalty, is 0.
    sub_pop = subset_feasible(population)

    len_subpop = len(sub_pop)

    # if no feasible individuals are found, keep the mutation rate high to explore the search space
    if sub_pop == []:
        if verbose is True:
            print("Exploring for feasible individuals. Mutpb: {}, Cxpb:{}".format(mutpb, cxpb))
    else:
        # if feasible individuals are found, start decreasing the mutation rate and increasing the crossover rate, to improve exploitation
        if cxpb < cx_lim:
            mutpb = mutpb - 0.01
            cxpb = cxpb + 0.01

        if verbose is True:
            print("{}/{} ({}%) FEASIBLE INDIVIDUALS".format(len_subpop, len(population),
                                                        round(len(sub_pop) / len(population) * 100, 2)))
            print("Mutpb: {}, Cxpb:{}".format(mutpb, cxpb))
            print("\n")

    np.random.shuffle(good_indexes_original)
    good_indexes = copy(good_indexes_original)
    good_list = copy(good_indexes)

    while len(offspring) < lambda_:
        op_choice = random.random()
        if op_choice < cxpb:                                                                        # Apply crossover
            # selection of 2 different niches for crossover. The search is performed in the 3d space
            cat = np.zeros((2))
            idxs = np.zeros((2,3))
            for i in range(2):
                if len(good_list) == 0:
                    good_list = copy(good_indexes)
                used = random.choice(range(len(good_list)))
                cat[i] = used
                idxs[i] = good_list[used,:]
                good_list = np.delete(good_list, used, 0)
            # select random individuals from first and second niches
            ind1, ind2 = map(toolbox.clone, [random.choice(sub_div["cat{}_{}_{}".format(int(idxs[0][0]), int(idxs[0][1]), int(idxs[0][2]))]),
                                             random.choice(sub_div["cat{}_{}_{}".format(int(idxs[1][0]), int(idxs[1][1]), int(idxs[1][2]))])])
            tries = 0
            while sum(ind1.fitness.values) == sum(ind2.fitness.values) and tries < 10: # if the same individual is selected, then repeat the niches selection
                                                                                       # process for a maximum of 10 times
                # selection of 2 different niches for crossover
                cat = np.zeros((2))
                idxs = np.zeros((2, 3))
                for i in range(2):
                    if len(good_list) == 0:
                        good_list = copy(good_indexes)
                    used = random.choice(range(len(good_list)))
                    cat[i] = used
                    idxs[i] = good_list[used, :]
                    good_list = np.delete(good_list, used, 0)
                # select random individuals from first and second niches
                ind1, ind2 = map(toolbox.clone, [
                    random.choice(sub_div["cat{}_{}_{}".format(int(idxs[0][0]), int(idxs[0][1]), int(idxs[0][2]))]),
                    random.choice(sub_div["cat{}_{}_{}".format(int(idxs[1][0]), int(idxs[1][1]), int(idxs[1][2]))])])
                tries += 1
            # perform crossover on selected individuals
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
            if len(offspring) < lambda_:
                del ind2.fitness.values
                offspring.append(ind2)
        else:                                                                                       # Apply mutation
            if len(good_list) == 0:
                good_list = copy(good_indexes)
            used = random.choice(range(len(good_list)))
            idxs = good_list[used, :]
            good_list = np.delete(good_list, used, 0)
            ind = toolbox.clone(random.choice(sub_div["cat{}_{}_{}".format(int(idxs[0]), int(idxs[1]), int(idxs[2]))]))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)

    return offspring, len_subpop, mutpb, cxpb


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """
    This function is a modification of the VarOr function implemented in the DEAP library
    (https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L192)

    The modification consists in:
        - use a while loop instead of a for loop
        - pass the best of the 2 children produced during crossover
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    while len(offspring) < lambda_:
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = deepcopy(toolbox.mate(ind1, ind2))
            if sum(ind1.fitness.wvalues) > sum(ind2.fitness.wvalues):
                del ind1.fitness.values
                offspring.append(ind1)
            else:
                del ind2.fitness.values
                offspring.append(ind2)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = deepcopy(toolbox.mutate(ind))
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring
