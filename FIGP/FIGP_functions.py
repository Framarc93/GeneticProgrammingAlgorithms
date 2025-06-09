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
File containing the modification introduced into the DEAP library and new functions and classes developed for the
Full Inclusive Genetic Programming algorithm [3]

References:
[1] Entropy-Driven Adaptive Representation. J. P. Rosca. Proceedings of the Workshop on Genetic Programming: From Theory
to Real-World Applications, 23-32. 1995
[2] https://pastebin.com/QKMhafRq
[3] F. Marchetti, M. Castelli, I. Bakurov and L. Vanneschi, "Full Inclusive Genetic Programming," 2024 IEEE Congress
on Evolutionary Computation (CEC), Yokohama, Japan, 2024, pp. 1-8, doi: 10.1109/CEC60901.2024.10611808.
"""

import sys
sys.path.append("..")
import numpy as np
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from deap import tools, gp
from functools import partial
import dill
from IGP.IGP_functions import subset_feasible
import random

#######################################################################################################################
"""                                         NEW FUNCTIONS AND CLASSES                                               """
#######################################################################################################################


##################### Population class
class POP_pheno_3D_2fit(object):
    '''
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    This class is used to collect data about a population. Used at the beginning for the selection of the initial
        population. Entropy measure comes from [1]. This class is used to evaluate different parameters regarding
        the population
    Attributes:
        items : list
            population to which apply the POP class
        lens : float
            array containing the lengths of each individual in the population
        max : int
            maximum length of the individuals in the population
        min : int
            minimum length of the individuals in the population
        maxDiff : int
            difference in length between the biggest and the smallest individual
        categories : dict
            dictionary containing the different categories in which the population is divided
        indexes : int
            array contatining the indexes of the filled categories
        entropy : float
            entropy measure of the population calculated according to [1]
    Methods:
        output_stats():
            print the statistics of the considered population
    '''

    def __init__(self, population, creator, cat_number_len, cat_number_fit, cat_number_height, fit_scale):
        self.items = list()
        self.lens = np.zeros(len(population))
        self.fits = np.zeros(len(population))
        self.fits_val = np.zeros(len(population))
        self.entropy = 0
        for i in range(len(population)):
            item = deepcopy(population[i])
            self.items.append(item)
            self.lens[i] = len(population[i])
            self.fits[i] = population[i].fitness.values[0]
            self.fits_val[i] = population[i].fitness_validation.values[0]
        self.min_len = int(min(self.lens))
        self.max_len = int(max(self.lens))

        self.min_fit = min(self.fits)
        self.max_fit = max(self.fits)

        self.min_fit_val = min(self.fits_val)
        self.max_fit_val = max(self.fits_val)

        self.categories, self.indexes = subset_diversity_pheno3D_2fit(population, creator, cat_number_len, cat_number_fit, cat_number_height, fit_scale)
        pp_fit = self.categories["distribution"]["percentage"]
        pp_fit = pp_fit[pp_fit != 0]
        self.entropy = -sum(pp_fit * np.log(pp_fit))

    def output_stats(self, Title, cat_number_len, cat_number_fit):
        print("\n")
        print("--------------------------- {} STATISTICS --------------------------------".format(Title))
        print("-- Min fit: {}, Max fit: {}, Min len: {}, Max len: {}, Min Fit Val: {}, Max Fit Val: {}---".format(self.min_fit,
                                                                                                                self.max_fit, self.min_len, self.max_len,
                                                                                                                self.min_fit_val, self.max_fit_val))
        print("-- Entropy: {0:.3f} -----------------------------".format(self.entropy))
        #print("-- Distribution : --")
        #print(tabulate(np.reshape(self.categories["distribution"]["percentage"] * 100, (cat_number_fit, cat_number_len))))

    def save_stats(self, data, path, gen):
        data = np.vstack((data, [self.min_fit, self.max_fit, self.min_len, self.max_len, self.entropy,
                                 self.categories]))
        np.save(path + 'Population_statistics_gen_{}'.format(gen), data)
        np.save(path + 'Individuals_lengths_gen_{}'.format(gen), self.lens)

    def retrieve_stats(self, data, lengths):
        data = np.vstack((data, np.array(([self.min_len, self.max_len, self.entropy,
                                           self.categories["distribution"]["percentage"]]), dtype=object)))
        lengths.append(self.lens)
        return data, lengths

    def plot_entropy_evol(self, save_path, gen, cat_len, cat_fit, cat_fit_val, fit_scale):
        #plt.figure(figsize=(5, 5))
        #len_range = np.linspace(self.min_len, self.max_len, cat_len+1)
        #fit_range = np.geomspace(self.min_fit, self.max_fit*fit_scale, cat_fit+1)
        #plt.vlines(x=len_range, ymin=self.min_fit, ymax=self.max_fit*fit_scale, colors='k')
        #plt.hlines(y=fit_range, xmin=self.min_len, xmax=self.max_len, colors='k')
        #plt.title('Gen {}'.format(gen))
        #plt.xlabel('Number of nodes')
        #plt.ylabel('Fitness')
        #count = 0
        #for i in range(len(fit_range)-1):
        #    for j in range(len(len_range)-1):
        #        plt.plot((len_range[j]+len_range[j+1])/2, (fit_range[i]+fit_range[i+1])/2, 'o', markersize=self.categories['distribution']['percentage'][count]*100, color='r')
        #        count += 1

        ax = plt.figure().add_subplot(projection='3d')
        plt.rcParams["figure.figsize"] = [6, 6]

        ax.azim = -134  # y rotation (default=270)
        ax.elev = 38  # x rotation (default=0)

        q3 = np.quantile(self.fits, fit_scale)
        q3_val = np.quantile(self.fits_val, fit_scale)
        len_range = np.linspace(self.min_len, self.max_len, cat_len + 1)
        fit_range = np.geomspace(self.min_fit, q3, cat_fit + 1)
        fit_val_range = np.geomspace(self.min_fit_val, q3_val, cat_fit_val + 1)

        for fit_val in fit_val_range:
            for fit in fit_range:
                ax.plot([fit_val, fit_val], [self.min_len, self.max_len], [fit, fit], color='k', linewidth=0.3, alpha=0.3)
        for fit_val in fit_val_range:
            for Len in len_range:
                ax.plot([fit_val, fit_val], [Len, Len], [self.min_fit, q3], color='k', linewidth=0.3, alpha=0.3)
        for fit in fit_range:
            for Len in len_range:
                ax.plot( [self.min_fit_val, q3_val], [Len, Len], [fit, fit], color='k', linewidth=0.3, alpha=0.3)
        count = 0
        for i in range(len(fit_range)-1):
            for j in range(len(len_range)-1):
                for k in range(len(fit_val_range)-1):
                    ax.plot((fit_val_range[k]+fit_val_range[k+1])/2, (len_range[j]+len_range[j+1])/2, (fit_range[i]+fit_range[i+1])/2, 'o', markersize=self.categories['distribution']['percentage'][count]*150, color='r')
                    count += 1
        plt.title('Gen {}'.format(gen))
        ax.set_xlabel('Validation Fitness')
        ax.set_ylabel('Length')
        ax.set_zlabel('Training Fitness')

        #plt.show()
        plt.savefig(save_path + '/gen_{}.jpg'.format(gen), format='jpg')
        plt.close()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __str__(self):
        return str(self.items)


##################### Niches generation mechanism
def subset_diversity_pheno3D_2fit(population, creator, cat_number_len, cat_number_fit, cat_number_height, fit_scale):
    """
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    Function used to divide the individuals in the input population into cat_number categories. The division into
    categories is done according to the training fitness, validation fitness and length of the individuals
    in the population
    """

    fits = []
    categories = {}
    distribution = []
    distr_stats = {}
    useful_ind = []
    lens = []
    fits_val = []
    for ind in population:
        fits.append(ind.fitness.values[0])
        fits_val.append(ind.fitness_validation.values[0])
        lens.append(len(ind))

    fits = np.array((fits))
    fits_val = np.array((fits_val))
    lens = np.array((lens))

    upper_fit = np.quantile(fits, fit_scale)
    upper_fit_val = np.quantile(fits_val, fit_scale)

    int_lens = np.linspace(min(lens), max(lens), cat_number_len+1)
    int_fits = np.geomspace(min(fits), upper_fit, cat_number_fit+1)
    int_fits_val = np.geomspace(min(fits_val), upper_fit_val, cat_number_height+1)

    for i in range(cat_number_fit):
        for j in range(cat_number_len):
            for k in range(cat_number_height):
                categories["cat{}_{}_{}".format(i, j, k)] = []

    for ind in population:
        break_out_flag = False
        fit = ind.fitness.values[0]
        fit_val = ind.fitness_validation.values[0]
        ind_len = len(ind)
        for i in range(cat_number_fit):
            if i == cat_number_fit - 1:
                for j in range(cat_number_len):
                    if ind_len >= int_lens[j] and ind_len <= int_lens[j + 1]:
                        for k in range(cat_number_height):
                            if k == cat_number_height - 1:
                                categories['cat{}_{}_{}'.format(i, j, k)].append(ind)
                                break_out_flag = True
                                break
                            elif fit_val >= int_fits_val[k] and fit_val <= int_fits_val[k + 1]:
                                categories['cat{}_{}_{}'.format(i, j, k)].append(ind)
                                break_out_flag = True
                                break
                        if break_out_flag is True:
                            break
                if break_out_flag is True:
                    break
            elif fit >= int_fits[i] and fit <= int_fits[i+1]:
                for j in range(cat_number_len):
                    if ind_len >= int_lens[j] and ind_len <= int_lens[j+1]:
                        for k in range(cat_number_height):
                            if k == cat_number_height - 1:
                                categories['cat{}_{}_{}'.format(i, j, k)].append(ind)
                                break_out_flag = True
                                break
                            elif fit_val >= int_fits_val[k] and fit_val <= int_fits_val[k + 1]:
                                categories['cat{}_{}_{}'.format(i, j, k)].append(ind)
                                break_out_flag = True
                                break
                        if break_out_flag is True:
                            break
                if break_out_flag is True:
                    break

    for i in range(cat_number_fit):
        for j in range(cat_number_len):
            for k in range(cat_number_height):
                if categories["cat{}_{}_{}".format(i, j, k)] != []:
                    useful_ind.append([i, j, k])
                distribution.append(len(categories["cat{}_{}_{}".format(i, j, k)]))

    distr_stats["individuals"] = distribution
    distr_stats["percentage"] = np.array(distribution) / len(population)
    categories["distribution"] = distr_stats

    return categories, np.asarray(useful_ind, dtype=int)


#######################################################################################################################
"""                                   MODIFIED FUNCTIONS FROM DEAP LIBRARY                                          """
#######################################################################################################################


##################### Evolutionary strategy
def eaMuPlusLambdaTol_pheno_2fit(population, toolbox, mu, lambda_, ngen, cxpb, mutpb, pset, creator,
                      stats=None, halloffame=None, verbose=__debug__, **kwargs):
    """
    Modification of eaMuPlusLambda function from DEAP library, used by FIGP. Modifications include:
        - use of tolerance value for the first fitness function below which the evolution is stopped
        - added population class
        - added best individual selection
        - added possibility to save population and best ind at each gen
        - added subset definition function
        - use of modified VarOr algorithm
        - the best individual is always passed to the new generation

    Please refer to the original function in the DEAP library for the full description https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L248C1-L248C5.
    """

    gen = 0 # initialize generation

    # retrieve niches params
    cat_number_len = kwargs['cat_number_len']
    cat_number_fit = kwargs['cat_number_fit']
    cat_number_height = kwargs['cat_number_height']
    fit_scale = kwargs['fit_scale']

    # define logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Retrieve the individuals with an empty fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # perform fitness evaluation with or without multiprocessing
    fitnesses = toolbox.map(partial(toolbox.evaluate, pset=pset, compile=gp.compile, kwargs=kwargs), invalid_ind)

    # assign evaluated fitness to population
    invalid_ind_orig = [ind for ind in population if not ind.fitness.valid]
    for ind, fit in zip(invalid_ind_orig, fitnesses):
        ind.fitness.values = fit[:2]
        ind.fitness_validation.values = fit[2:]

    # compute statistics on population
    all_lengths = []
    data = np.array((['Min length', 'Max length', 'Entropy', 'Distribution']), dtype=object)
    pop = POP_pheno3D_2fit(population, creator, cat_number_len, cat_number_fit, cat_number_height, fit_scale)
    data, all_lengths = pop.retrieve_stats(data, all_lengths)

    # update hall of fame
    if halloffame is not None:
        halloffame.update(population, for_feasible=True)

    # update logbook
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # find min fitness
    min_fit = np.array(logbook.chapters["fitness"].select("min"))[-1][0]
    best_ind = copy(population[0])
    for i in range(len(population)):
        if population[i].fitness.values[0] == min_fit:
            best_ind = copy(population[i])

    # save data
    if kwargs['save_gen'] is not None:
        output = open(kwargs['save_path'] + 'Best_ind_{}'.format(gen), "wb")
        dill.dump(best_ind, output, -1)
        output.close()
        output = open(kwargs['save_path'] + 'Population_gen_{}'.format(gen), "wb")
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

        sub_div, good_index = subset_diversity_pheno3D_2fit(population, creator, cat_number_len, cat_number_fit, cat_number_height, fit_scale)
        offspring, len_feas, mutpb, cxpb = varOr_FIGP(population, toolbox, lambda_, sub_div, good_index, cxpb, mutpb,
                                                      verbose, kwargs['cx_lim'], False)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # evaluate fitness

        fitnesses = toolbox.map(partial(toolbox.evaluate, pset=pset, compile=gp.compile, kwargs=kwargs), invalid_ind)


        # assign fitness to evaluated individuals
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[:2]
            ind.fitness_validation.values = fit[2:]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(invalid_ind, for_feasible=True)

        # select new population from parents and offspring
        global_pop = population + offspring
        best_ind = selBest_simple(global_pop, 1)
        pop_without_best = [ind for ind in global_pop if ind != best_ind[0]]

        # compute statistics on population
        organized_pop, good_indexes = subset_diversity_pheno3D_2fit(pop_without_best, creator, cat_number_len, cat_number_fit, cat_number_height, fit_scale)
        population = copy(toolbox.select(mu-1, organized_pop, good_indexes))
        population = population + best_ind
        pop = POP_pheno3D_2fit(population, creator, cat_number_len, cat_number_fit, cat_number_height, fit_scale)
        data, all_lengths = pop.retrieve_stats(data, all_lengths)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        min_fit = np.array(logbook.chapters["fitness"].select("min"))[-1][0]
        best_ind = copy(population[0])
        for i in range(len(population)):
            if population[i].fitness.values[0] == min_fit:
                best_ind = copy(population[i])
        if kwargs['fit_tol'] is not None:
            if min_fit <= kwargs['fit_tol']:
                success = True
            else:
                success = False

        if kwargs['save_gen'] is not None:
            output = open(kwargs['save_path'] + 'Best_ind_{}'.format(gen), "wb")
            dill.dump(best_ind, output, -1)
            output.close()
            output = open(kwargs['save_path'] + 'Population_gen_{}'.format(gen), "wb")
            dill.dump(population, output, -1)
            output.close()

        gen += 1

    return population, logbook, data, all_lengths

##################### Reproduction function
def varOr_FIGP(population, toolbox, lambda_, sub_div, good_indexes_original, cxpb, mutpb, verbose, cx_lim, select_best):
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

