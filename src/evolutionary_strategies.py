# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------------ Copyright (C) 2025 Francesco Marchetti -----------------
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

from deap import tools
from functools import partial
import numpy as np
from copy import copy
import dill
from src.selection_functions import selBest_IGP
from deap.tools import selBest


def InclusiveMuPlusLambda(population, toolbox, mu, lambda_, ngen, cxpb, mutpb, pset, creator, stats=None,
                          halloffame=None, verbose=__debug__, **kwargs):
    """
    Modification of eaMuPlusLambda function from DEAP library, used by IGP. Modifications include:
        - use of tolerance value for the first fitness function below which the evolution is stopped
        - added population class
        - added best individual selection
        - added possibility to save population and best ind at each gen
        - added subset definition function
        - use of modified VarOr algorithm
        - the best individual is always passed to the new generation

    Please refer to the original function in the DEAP library for the full description
    https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L248C1-L248C5.
    """

    kwargs = kwargs['kwargs']

    gen = 0  # initialize generation

    # define logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Retrieve the individuals with an empty fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # perform fitness evaluation with or without multiprocessing
    fitnesses = toolbox.map(partial(toolbox.evaluate, compile=toolbox.compile, kwargs=kwargs), invalid_ind)

    # assign evaluated fitness to population
    invalid_ind_orig = [ind for ind in population if not ind.fitness.valid]
    for ind, fit in zip(invalid_ind_orig, fitnesses):
        ind.fitness.values = fit[:2]
        if len(fit) > 2:
            ind.fitness_validation.values = fit[2:]

    # compute statistics on population
    all_lengths = []
    data = np.array((['Min length', 'Max length', 'Entropy', 'Distribution']), dtype=object)
    pop = toolbox.POP_class(population, creator, kwargs=kwargs)
    data, all_lengths = pop.retrieve_stats(data, all_lengths)

    # update hall of fame
    if halloffame is not None:
        halloffame.update(population, for_feasible=True)

    # update logbook
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # find min fitness
    min_fit = np.array(logbook.chapters["fitness"].select("min"))[-1][0]
    best_ind = copy(population[0])
    for i in range(len(population)):
        if population[i].fitness.values[0] == min_fit:
            best_ind = copy(population[i])

    # save data
    if (kwargs['save_gen'] is not None) and (kwargs['save_gen'] is True):
        output = open(kwargs['save_path'] + 'Best_ind_{}'.format(gen), "wb")
        dill.dump(best_ind, output, -1)
        output.close()

    if (kwargs['save_pop'] is not None) and (kwargs['save_pop'] is True):
        output = open(kwargs['save_path'] + 'Full_population_{}'.format(gen), "wb")
        dill.dump(population, output, -1)
        output.close()

    # check stopping criterion
    success = False
    if kwargs['fit_tol'] is not None:
        if min_fit <= kwargs['fit_tol']:
            success = True
        else:
            success = False

    # Begin the generational process
    gen += 1

    while gen < ngen and not success:

        # create niches on population
        sub_div, good_index = toolbox.niches_generation(population, creator, kwargs=kwargs)

        # Perform crossover, mutation and pass
        offspring, len_feas, mutpb, cxpb = toolbox.varOr(population, toolbox, lambda_, sub_div, good_index, cxpb, mutpb,
                                                     verbose, kwargs['cx_lim'])

        # Retrieve the individuals with an empty fitness from the offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # evaluate fitness
        fitnesses = toolbox.map(partial(toolbox.evaluate, compile=toolbox.compile, kwargs=kwargs), invalid_ind)

        # assign fitness to evaluated individuals
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[:2]
            if len(fit) > 2:
                ind.fitness_validation.values = fit[2:]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(invalid_ind, for_feasible=True)

        # create total population
        global_pop = population + offspring
        # retrieve best individuals
        best_ind = selBest_IGP(global_pop)
        # remove best individual from total population. This will be passed anyway to the next generation
        pop_without_best = [ind for ind in global_pop if ind != best_ind]

        # Create niches on total population
        organized_pop, good_indexes = toolbox.niches_generation(pop_without_best, creator, kwargs=kwargs)

        # perform selection
        population = copy(toolbox.select(mu - 1, organized_pop, good_indexes))

        # add back best individual
        population = population + [best_ind]

        # compute statistics on population
        pop = toolbox.POP_class(population, creator, kwargs=kwargs)
        data, all_lengths = pop.retrieve_stats(data, all_lengths)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Update best individual
        min_fit = np.array(logbook.chapters["fitness"].select("min"))[-1][0]
        best_ind = copy(population[0])
        for i in range(len(population)):
            if population[i].fitness.values[0] == min_fit:
                best_ind = copy(population[i])

        # Check if fitness is below tolerance
        if kwargs['fit_tol'] is not None:
            if min_fit <= kwargs['fit_tol']:
                success = True
            else:
                success = False

        # Save data
        if (kwargs['save_gen'] is not None) and (kwargs['save_gen'] is True):
            output = open(kwargs['save_path'] + 'Best_ind_{}'.format(gen), "wb")
            dill.dump(best_ind, output, -1)
            output.close()

        if (kwargs['save_pop'] is not None) and (kwargs['save_pop'] is True):
            output = open(kwargs['save_path'] + 'Full_population_{}'.format(gen), "wb")
            dill.dump(population, output, -1)
            output.close()

        gen += 1

    return population, logbook, data, all_lengths, halloffame


def MuPlusLambdaMGGP(population, toolbox, mu, lambda_, ngen, cxpb, mutpb, pset, creator,
                      stats=None, halloffame=None, verbose=__debug__, **kwargs):

    """
        Modification of eaMuPlusLambda function from DEAP library, used by MGGP. Modifications include:
            - use of tolerance value for the first fitness function below which the evolution is stopped
            - added population class
            - added best individual selection
            - added possibility to save population and best ind at each gen
            - the best individual is always passed to the new generation
            - it considers the linear combination parameters needed by the MGGP in the fitness assignment

        Please refer to the original function in the DEAP library for the full description
        https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L248C1-L248C5.
        """

    kwargs = kwargs['kwargs']

    gen = 0  # initialize generation

    # define logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Retrieve the individuals with an empty fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # perform fitness evaluation with or without multiprocessing
    fitnesses = toolbox.map(partial(toolbox.evaluate, compile=toolbox.compile, kwargs=kwargs), invalid_ind)

    # assign evaluated fitness to population
    invalid_ind_orig = [ind for ind in population if not ind.fitness.valid]
    for ind, fit in zip(invalid_ind_orig, fitnesses):
        ind.fitness.values = fit[0]
        if len(fit) > 1:
            ind.fitness_validation.values = fit[2]
        ind.w = fit[-1]

    # compute statistics on population
    all_lengths = []
    data = np.array((['Min length', 'Max length', 'Entropy', 'Distribution']), dtype=object)
    pop = toolbox.POP_class(population, creator, kwargs=kwargs)
    data, all_lengths = pop.retrieve_stats(data, all_lengths)

    # update hall of fame
    if halloffame is not None:
        halloffame.update(population, for_feasible=True)

    # update logbook
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    # find min fitness
    min_fit = np.array(logbook.chapters["fitness"].select("min"))[-1][0]
    best_ind = copy(population[0])
    for i in range(len(population)):
        if population[i].fitness.values[0] == min_fit:
            best_ind = copy(population[i])

    # save data
    if (kwargs['save_gen'] is not None) and (kwargs['save_gen'] is True):
        output = open(kwargs['save_path'] + 'Best_ind_{}'.format(gen), "wb")
        dill.dump(best_ind, output, -1)
        output.close()

    if (kwargs['save_pop'] is not None) and (kwargs['save_pop'] is True):
        output = open(kwargs['save_path'] + 'Full_population_{}'.format(gen), "wb")
        dill.dump(population, output, -1)
        output.close()

    # check stopping criterion
    success = False
    if kwargs['fit_tol'] is not None:
        if min_fit <= kwargs['fit_tol']:
            success = True
        else:
            success = False

    # Begin the generational process
    gen += 1

    while gen < ngen and not success:

        # create niches on population
        offspring = toolbox.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Retrieve the individuals with an empty fitness from the offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # evaluate fitness
        fitnesses = toolbox.map(partial(toolbox.evaluate, compile=toolbox.compile, kwargs=kwargs), invalid_ind)

        # assign fitness to evaluated individuals
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0]
            if len(fit) > 1:
                ind.fitness_validation.values = fit[2]
            ind.w = fit[-1]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring, for_feasible=True)

        # create total population
        global_pop = population + offspring
        # retrieve best individuals
        best_ind = selBest(global_pop, 1)[0]
        # remove best individual from total population. This will be passed anyway to the next generation
        pop_without_best = [ind for ind in global_pop if ind != best_ind]

        population = copy(toolbox.select(pop_without_best, mu - 1))

        # add back best individual
        population = population + [best_ind]

        # compute statistics on population
        pop = toolbox.POP_class(population, creator, kwargs=kwargs)
        data, all_lengths = pop.retrieve_stats(data, all_lengths)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)
        if verbose:
            print(logbook.stream)

        # Update best individual
        min_fit = np.array(logbook.chapters["fitness"].select("min"))[-1][0]
        best_ind = copy(population[0])
        for i in range(len(population)):
            if population[i].fitness.values[0] == min_fit:
                best_ind = copy(population[i])

        # Check if fitness is below tolerance
        if kwargs['fit_tol'] is not None:
            if min_fit <= kwargs['fit_tol']:
                success = True
            else:
                success = False

        # Save data
        if (kwargs['save_gen'] is not None) and (kwargs['save_gen'] is True):
            output = open(kwargs['save_path'] + 'Best_ind_{}'.format(gen), "wb")
            dill.dump(best_ind, output, -1)
            output.close()

        if (kwargs['save_pop'] is not None) and (kwargs['save_pop'] is True):
            output = open(kwargs['save_path'] + 'Full_population_{}'.format(gen), "wb")
            dill.dump(population, output, -1)
            output.close()

        gen += 1

    return population, logbook, data, all_lengths, halloffame