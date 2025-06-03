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

"""
File containing the modification introduced into the DEAP library and new functions and classes developed for the Inclusive Genetic Programmign algorithm [3]

References:
[1] Entropy-Driven Adaptive Representation. J. P. Rosca. Proceedings of the Workshop on Genetic Programming: From Theory
to Real-World Applications, 23-32. 1995
[2] https://pastebin.com/QKMhafRq
[3] Marchetti, F., Minisci, E. (2021). Inclusive Genetic Programming. In: Hu, T., Lourenço, N., Medvet, E. (eds) Genetic Programming. EuroGP 2021. 
    Lecture Notes in Computer Science(), vol 12691. Springer, Cham. https://doi.org/10.1007/978-3-030-72812-0_4
"""
import numpy as np
import dill
from copy import deepcopy, copy
import random
from functools import partial, wraps
from deap import tools, gp
from deap.tools import selBest
from operator import eq




#######################################################################################################################
"""                                         NEW FUNCTIONS AND CLASSES                                               """
#######################################################################################################################

##################### Selection mechanisms
def InclusiveTournament(mu, organized_pop, good_indexes, selected_individuals, fitness_size, parsimony_size, creator):
    """
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    Rationale behind InclusiveTournament: a double tournament selection is performed in each category, so to maintain
    diversity. Double Tournament is used so to avoid bloat. An exploited measure is used to point out when a category is
    completely exploited. For example, if in a category are present only 4 individuals, the tournament will be
    performed at maximum 4 times in that category. This to avoid a spreading of clones of the same individuals.
    """

    chosen = []
    exploited = np.zeros((len(good_indexes)))
    j = 0
    while len(chosen) < mu:
        if j > len(good_indexes) - 1:
            j = 0
        i = good_indexes[j]

        if exploited[j] < len(organized_pop["cat{}".format(i)]):
            if len(organized_pop["cat{}".format(i)]) > 1:
                selected = selDoubleTournament_IGP(organized_pop["cat{}".format(i)], selected_individuals, fitness_size,
                                               parsimony_size, creator, fitness_first=True)
                chosen.append(selected)
            else:
                chosen.append(organized_pop["cat{}".format(i)][0])
            exploited[j] += 1
        j += 1
    return chosen


def selBest_IGP(individuals):
    """
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    This function selects the best individuals in a population composed by individuals with multiple trees and whose fitness array 
    is composed by at least two elements. The first element is the main fitness function driving the evolutionary process. 
    The second fitness is a penalty term used if constraints are applied to the problem.

    The logic behind this function is to always select feasible individuals (last fitness term = 0). If both individuals are 
    feasible then select the best one (either with bigger or smaller fitness according to the problem). If both individuals are unfeasible, 
    then 10% of the times, select the one with the lowest penalty, while 90% ot the times, select the one with the best sum of all penalties.

    """
    best = individuals[0]
    choice = random.random()
    for ind in individuals:                                                    
        if ind.fitness.values[-1] == 0 and best.fitness.values[-1] == 0:       
            if ind.fitness.wvalues[0] > best.fitness.wvalues[0]:               
                best = ind                                                     
        elif ind.fitness.values[-1] == 0 and best.fitness.values[-1] != 0:     
            best = ind                                                         
        elif ind.fitness.values[-1] != 0 and best.fitness.values[-1] != 0:     
            if choice > 0.9:                                                   
                if ind.fitness.wvalues[-1] > best.fitness.wvalues[-1]:         
                    best = ind                                                 
            else:                                                              
                if sum(ind.fitness.wvalues) > sum(best.fitness.wvalues):       
                    best = ind                                                 
    return best

##################### Population class
class POP_geno(object):
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

    def __init__(self, population, creator):
        self.items = list()
        self.lens = np.zeros(len(population))
        self.fits = np.zeros(len(population))
        for i in range(len(population)):
            item = deepcopy(population[i])
            self.items.append(item)
            if hasattr(creator, 'SubIndividual'):
                self.lens[i] = sum([len(gpt) for gpt in item])
            else:
                self.lens[i] = len(population[i])
        self.min_len = int(min(self.lens))
        self.max_len = int(max(self.lens))
        self.maxDiff_len = self.max_len - self.min_len
        self.categories_len, self.indexes_len = subset_diversity_genotype(population, creator)
        self.categories_fit = copy(self.categories_len)
        pp_len = self.categories_len["distribution"]["percentage"]
        pp_len = pp_len[pp_len != 0]
        self.entropy_len = -sum(pp_len * np.log(pp_len))


    def output_stats(self, Title):
        print("\n")
        print("--------------------------- {} STATISTICS --------------------------------".format(Title))
        print("-- Min len: {}, Max len: {} ---".format(self.min_len, self.max_len))
        print("-- Entropy len: {0:.3f} -----------------------------".format(self.entropy_len))
        print("-- Distribution len: {} ------------------------------".format(self.categories_len["distribution"]["percentage"] * 100))
        print("----------------------------------------------------------------------------")

    def save_stats(self, data, path, gen):
        data = np.vstack((data, [self.min_len, self.max_len, self.entropy_len, self.categories_len["distribution"]["percentage"]]))
        np.save(path + 'Population_statistics_gen_{}'.format(gen), data)
        np.save(path + 'Individuals_lengths_gen_{}'.format(gen), self.lens)

    def retrieve_stats(self, data, lengths):
        data = np.vstack((data, np.array(([self.min_len, self.max_len, self.entropy_len, self.categories_len["distribution"]["percentage"]]), dtype=object)))
        lengths.append(self.lens)
        return data, lengths

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


##################### Niches creation
def subset_feasible(population):
    """
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    Function used to create a subset of feasible individuals from the input population"""
    sub_pop = []
    for ind in population:
        if ind.fitness.values[-1] == 0:
            sub_pop.append(ind)
    return sub_pop


def subset_diversity_genotype(population, creator):
    """
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    Function used to divide the individuals in the input population into cat_number categories. The division into
    categories is done according to the length of the individuals in the population"""
    cat_number = 10  # here the number of categories is selected
    lens = []
    categories = {}
    distribution = []
    distr_stats = {}
    invalid_ind = []

    for ind in population:
        try:
            if type(ind[0]) == creator.SubIndividual and len(ind) > 1:
                lens.append((sum([len(gpt) for gpt in ind])))
            else:
                lens.append(len(ind))
        except AttributeError:
            lens.append(len(ind))
    cat = np.linspace(min(lens), max(lens), cat_number + 1)
    useful_ind = np.linspace(0, len(cat) - 2, len(cat) - 1)

    for i in range(len(cat) - 1):
        categories["cat{}".format(i)] = []
    for ind in population:
        for i in range(len(cat) - 1):
            try:
                if type(ind[0]) == creator.SubIndividual:
                    totLen = sum([len(gpt) for gpt in ind])
                else:
                    totLen = len(ind)
            except AttributeError:
                totLen = len(ind)
            if totLen >= cat[i] and totLen <= cat[i + 1]:
                categories["cat{}".format(i)].append(ind)
                break

    for i in range(len(cat) - 1):
        if categories["cat{}".format(i)] == []:
            invalid_ind.append(i)
        distribution.append(len(categories["cat{}".format(i)]))
    distr_stats["individuals"] = distribution
    distr_stats["percentage"] = np.array(distribution) / len(population)
    categories["distribution"] = distr_stats
    if invalid_ind != []:
        useful_ind = np.delete(useful_ind, invalid_ind, 0)
    return categories, np.asarray(useful_ind, dtype=int)




#######################################################################################################################
"""                                   MODIFIED FUNCTIONS FROM DEAP LIBRARY                                          """
#######################################################################################################################

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
    # retrive subset of feasible individuals. Feasible means that the second fitness, i.e. the penalty, is 0.
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
        if op_choice < cxpb:                                                                        # Apply crossover
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
            while sum(ind1.fitness.values) == sum(ind2.fitness.values) and tries < 10: # if the same individual is selected, then repeat the niches selection 
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
            # peform crossover on selected individuals
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
            if len(offspring) < lambda_:
                del ind2.fitness.values
                offspring.append(ind2)
        else:                                                                                        # Apply mutation
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

############## MODIFIED BLOAT CONTROL #########################################
def staticLimitMod(key, max_value):
    """This is a modification of the staticLimit function implemented in the DEAP library in order to
    deal with individual composed by multiple trees

    Original description:
    Implement a static limit on some measurement on a GP tree, as defined
    by Koza in [Koza1989]. It may be used to decorate both crossover and
    mutation operators. When an invalid (over the limit) child is generated,
    it is simply replaced by one of its parents, randomly selected.

    This operator can be used to avoid memory errors occuring when the tree
    gets higher than 90 levels (as Python puts a limit on the call stack
    depth), because it can ensure that no tree higher than this limit will ever
    be accepted in the population, except if it was generated at initialization
    time.

    :param key: The function to use in order the get the wanted value. For
                instance, on a GP tree, ``operator.attrgetter('height')`` may
                be used to set a depth limit, and ``len`` to set a size limit.
    :param max_value: The maximum value allowed for the given measurement.
    :returns: A decorator that can be applied to a GP operator using \
    :func:`~deap.base.Toolbox.decorate`

    .. note::
       If you want to reproduce the exact behavior intended by Koza, set
       *key* to ``operator.attrgetter('height')`` and *max_value* to 17.

    .. [Koza1989] J.R. Koza, Genetic Programming - On the Programming of
        Computers by Means of Natural Selection (MIT Press,
        Cambridge, MA, 1992)

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                check = max(ind, key=key)  # from here modified in order to consider an individual
                for k in range(len(ind)):
                    if ind[k] == check:
                        j = copy(k)
                if key(check) > max_value:
                    new_inds[i][j] = random.choice(keep_inds)[j]
            return new_inds

        return wrapper

    return decorator


######################## MODIFIED SELECTION MECHANISMS ###########################

def selDoubleTournament_IGP(individuals, k, fitness_size, parsimony_size, creator, fitness_first):
    """This is a modification of the xselDoubleTournament function from [2] which itself is a modification of
     the selDoubleTournament function implemented in the DEAP library (https://github.com/DEAP/deap/blob/master/deap/tools/selection.py#L106). 
     
     The modification is done in order to deal with individual composed by multiple trees and with the penalty fitness used by the IGP.

    """
    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, k, select):
        """
        From [2] and modified to check for SubIndividual attribute
        """
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            if len(individuals) == 1:
                return random.choice(individuals)
            else:
                prob = parsimony_size / 2.
                ind1, ind2 = select(individuals, k=2)

                if hasattr(creator, 'SubIndividual') and len(ind1) > 0:
                    lind1 = sum([len(gpt) for gpt in ind1])  
                    lind2 = sum([len(gpt) for gpt in ind2])  
                else:
                    lind1 = len(ind1)
                    lind2 = len(ind2)
                if lind1 > lind2:
                    ind1, ind2 = ind2, ind1
                elif lind1 == lind2:
                    # random selection in case of a tie
                    prob = 0.5

                # Since size1 <= size2 then ind1 is selected
                # with a probability prob
                chosen.append(ind1 if random.random() < prob else ind2)

            return chosen[0]

    def _fitTournament(individuals, k):
        """
        Author(s): Francesco Marchetti
        email: framarc93@gmail.com

        This fitTournament function is used to select the individual with the best fitness according to both its fitness and penalty values.
        The logic behind this function is to always select feasible individuals (last fitness term = 0). If both individuals are 
        feasible then select the best one (either with bigger or smaller fitness according to the problem). If both individuals are unfeasible, 
        then select the one with the lowest penalty. The penalty is always minimized.
        """
        chosen = []
        for _ in range(k):
            a1, a2 = random.sample(individuals, 2)
            if a1.fitness.values[-1] == 0 and a2.fitness.values[-1] == 0:
                if sum(a1.fitness.wvalues) > sum(a2.fitness.wvalues):
                    chosen.append(a1)
                else:
                    chosen.append(a2)
            elif a1.fitness.values[-1] == 0 and a2.fitness.values[-1] != 0:
                chosen.append(a1)
            elif a1.fitness.values[-1] != 0 and a2.fitness.values[-1] == 0:
                chosen.append(a2)
            elif a1.fitness.values[-1] != 0 and a2.fitness.values[-1] != 0:
                if a1.fitness.values[-1] < a2.fitness.values[-1]:
                    chosen.append(a1)
                else:
                    chosen.append(a2)

        return chosen

    if fitness_first:
        tfit = partial(_fitTournament)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, k)



####################### MODIFIED HALL OF FAME #############################

class HallOfFame_IGP(object):
    """
    Modified HallOfFame class taken from the DEAP library (https://github.com/DEAP/deap/blob/master/deap/tools/support.py#L488). 
    The introduced modifications allow for:
        - a inverted classification (the last individual is the best) in comparison with the original
        - individuals are inserted in the hall of fame according to the following scheme:
            1 - for_feasible is checked, if is True, the best individuals are those with a penalty = 0 (last
                fitness function) and the lowest first fitness function
                1.1 - the individuals with a penalty = 0 are prioritized and then are compared on the value of the first
                      fitness function; the one with the lowest value are inserted.
                1.2 - the individuals with a penalty !=0 are compared on the sum of both the fitness. The ones with the
                      lowest sum are inserted
            2 - if for_feasible is False, the sum of both fitness is considered and the ones with the lowest sum
                are inserted
    """


    def __init__(self, maxsize, similar=eq):
        self.maxsize = maxsize
        self.keys = list()
        self.items = list()
        self.similar = similar

    def shuffle(self):
        arr_start = deepcopy(self.items)
        arr_end = []
        while len(arr_start) > 0:
            ind = random.randint(0, len(arr_start) - 1)
            arr_end.append(arr_start[ind])
            arr_start.pop(ind)
        return arr_end

    def update(self, population, for_feasible):
        """Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        if len(self) == 0 and self.maxsize != 0 and len(population) > 0:
            # Working on an empty hall of fame is problematic for the
            # "for else"
            self.insert(population[0], for_feasible)

        if for_feasible is True:                                                                                 # Modified part
            for ind in population:

                if ind.fitness.values[-1] == 0.0:                                                                # Modified part
                    if self[0].fitness.values[-1] == 0.0:                                                        # Modified part
                        if sum(ind.fitness.wvalues) > sum(self[0].fitness.wvalues) or len(self) < self.maxsize:  # Modified part
                            for hofer in self:
                                # Loop through the hall of fame to check for any
                                # similar individual
                                if self.similar(ind, hofer):
                                    break
                            else:
                                # The individual is unique and strictly better than
                                # the worst
                                if len(self) >= self.maxsize:
                                    self.remove(0)                                                               # Modified part
                                self.insert(ind, for_feasible)
                    else:
                        for hofer in self:
                            # Loop through the hall of fame to check for any
                            # similar individual
                            if self.similar(ind, hofer):
                                break
                        else:
                            # The individual is unique and strictly better than
                            # the worst
                            if len(self) >= self.maxsize:
                                self.remove(0)
                            self.insert(ind, for_feasible)

                elif (sum(ind.fitness.values) < sum(self[0].fitness.values)) or len(self) < self.maxsize:        # Modified part
                    for hofer in self:
                        # Loop through the hall of fame to check for any
                        # similar individual
                        if self.similar(ind, hofer):
                            break
                    else:
                        # The individual is unique and strictly better than
                        # the worst
                        if len(self) >= self.maxsize:
                            self.remove(0)                                                                      # Modified part
                        self.insert(ind, for_feasible)

        else:                                                                                                   # Modified part
            for ind in population:                                                                              # Modified part
                if sum(ind.fitness.wvalues) > sum(self[0].fitness.wvalues) or len(self) < self.maxsize:
                    for hofer in self:
                        # Loop through the hall of fame to check for any
                        # similar individual
                        if self.similar(ind, hofer):
                            break
                    else:
                        # The individual is unique and strictly better than
                        # the worst
                        if len(self) >= self.maxsize:
                            self.remove(0)                                                                     # Modified part
                        self.insert(ind, for_feasible)

    def insert(self, item, for_feasible):
        """Insert a new individual in the hall of fame using the
        :func:`~bisect.bisect_right` function. The inserted individual is
        inserted on the right side of an equal individual. Inserting a new
        individual in the hall of fame also preserve the hall of fame's order.
        This method **does not** check for the size of the hall of fame, in a
        way that inserting a new individual in a full hall of fame will not
        remove the worst individual to maintain a constant size.

        :param item: The individual with a fitness attribute to insert in the
                     hall of fame.
        """

        def bisect_right(a, x, lo=0, hi=None):
            """Return the index where to insert item x in list a, assuming a is sorted.
            The return value i is such that all e in a[:i] have e <= x, and all e in
            a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
            insert just after the rightmost x already there.
            Optional args lo (default 0) and hi (default len(a)) bound the
            slice of a to be searched.
            """

            if lo < 0:
                raise ValueError('lo must be non-negative')
            if hi is None:
                hi = len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                '''must indentify 4 cases: if both are feasible, if the new is feasible and the one in the list is not, viceversa and if both are infeasible'''
                if for_feasible is True:
                    # 1st case: both are feasible
                    if x.values[-1] == 0 and a[mid].values[-1] == 0:
                        if sum(x.wvalues) > sum(a[mid].wvalues):
                            hi = mid
                        else:
                            lo = mid + 1
                    # 2nd case: value to insert is feasible, the one in the list is not
                    elif x.values[-1] == 0 and a[mid].values[-1] != 0:
                        hi = mid
                    # 3rd case: value to insert is not feasible, the one in the list is feasible
                    elif x.values[-1] != 0 and a[mid].values[-1] == 0:
                        lo = mid + 1
                    # 4th case: both are infeasible
                    elif x.values[-1] != 0 and a[mid].values[-1] != 0:
                        if x.values[-1] < a[mid].values[-1]:
                            hi = mid
                        else:
                            lo = mid + 1
                else:
                    if sum(x.wvalues) > sum(a[mid].wvalues):
                        hi = mid
                    else:
                        lo = mid + 1
            return lo

        item = deepcopy(item)
        i = bisect_right(self.keys, item.fitness)
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, item.fitness)

    def remove(self, index):
        """Remove the specified *index* from the hall of fame.

        :param index: An integer giving which item to remove.
        """
        del self.keys[len(self) - (index % len(self) + 1)]
        del self.items[index]

    def clear(self):
        """Clear the hall of fame."""
        del self.items[:]
        del self.keys[:]

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



##############################  MODIFIED EVOLUTIONARY STRATEGIES  ##############################

def eaMuPlusLambdaTol(population, toolbox, mu, lambda_, ngen, cxpb, mutpb, pset, creator,
                      stats=None, halloffame=None, verbose=__debug__, **kwargs):
    r"""
    Modification of eaMuPlusLambda function from DEAP library, used by IGP. Modifications include:
        - use of tolerance value for the first fitness function below which the evolution is stopped
        - added population class
        - added best individual selection
        - added possibility to save population and best ind at each gen
        - added subset defintion function
        - use of modifierd VarOr algorithm
        - the best individual is always passed to the new generation

    Please refer to the original function in the DEAP library for the full description https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L248C1-L248C5. 
    """
   
    gen = 0 # initialize generation

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

    # compute statistics on population
    all_lengths = []
    data = np.array((['Min length', 'Max length', 'Entropy', 'Distribution']), dtype=object)
    pop = POP_geno(population, creator)
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
    if (kwargs['save_gen'] is not None) and (kwargs['save_gen'] == True):
        output = open(kwargs['save_path'] + 'Best_ind_{}'.format(gen), "wb")
        dill.dump(best_ind, output, -1)
        output.close()

    if (kwargs['save_pop'] is not None) and (kwargs['save_pop'] == True):
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
        sub_div, good_index = subset_diversity_genotype(population, creator)

        # Perform crossover, mutation and pass
        offspring, len_feas, mutpb, cxpb = varOr_IGP(population, toolbox, lambda_, sub_div, good_index, cxpb, mutpb,
                                                    verbose, kwargs['cx_lim'])

        # Retrieve the individuals with an empty fitness from the offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # evaluate fitness
        fitnesses = toolbox.map(partial(toolbox.evaluate, pset=pset, compile=gp.compile, kwargs=kwargs), invalid_ind)

        # assign fitness to evaluated individuals
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[:2]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(invalid_ind, for_feasible=True)

        # create total population
        global_pop = population + offspring
        # retrieve best individuals
        best_ind = selBest(global_pop, 1)
        # remove best individual from total popluation. This will be passed anyway to the next generation
        pop_without_best = [ind for ind in global_pop if ind != best_ind[0]]

        # Create niches on total population
        organized_pop, good_indexes = subset_diversity_genotype(pop_without_best, creator)

        # perform selection
        population = copy(toolbox.select(mu - 1, organized_pop, good_indexes))

        # add back best individual
        population = population + best_ind

        # compute statistics on population
        pop = POP_geno(population, creator)
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
        if (kwargs['save_gen'] is not None) and (kwargs['save_gen'] == True):
            output = open(kwargs['save_path'] + 'Best_ind_{}'.format(gen), "wb")
            dill.dump(best_ind, output, -1)
            output.close()

        if (kwargs['save_pop'] is not None) and (kwargs['save_pop'] == True):
            output = open(kwargs['save_path'] + 'Full_population_{}'.format(gen), "wb")
            dill.dump(population, output, -1)
            output.close()

        gen += 1

    return population, logbook, data, all_lengths, halloffame


########################### GENETIC OPERATORS FOR MULTIPLE TREE OUTPUT   #####################################

def xmateMultiple(ind1, ind2):
    """From [2] and modified"""
    for i in range(len(ind1)):
        ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2


def xmutMultiple(ind, expr, unipb, shrpb, inspb, pset, creator):
    """From [2] and modified. Added several mutations possibilities."""
    choice = random.random()
    try:
        if type(ind[0]) == creator.SubIndividual and len(ind) > 1:
            for i in range(len(ind)):
                if choice < unipb:
                    indx1 = gp.mutUniform(ind[i], expr, pset=pset)
                    ind[i] = indx1[0]
                elif unipb <= choice < unipb + shrpb:
                    indx1 = gp.mutShrink(ind[i])
                    ind[i] = indx1[0]
                elif unipb + shrpb <= choice < unipb + shrpb + inspb:
                    indx1 = gp.mutInsert(ind[i], pset=pset)
                    ind[i] = indx1[0]
                else:
                    choice2 = random.random()
                    if choice2 < 0.5:
                        indx1 = gp.mutEphemeral(ind[i], "all")
                        ind[i] = indx1[0]
                    else:
                        indx1 = gp.mutEphemeral(ind[i], "one")
                        ind[i] = indx1[0]
    except AttributeError:
        if choice < unipb:
            indx1 = gp.mutUniform(ind, expr, pset=pset)
            ind = indx1[0]
        elif unipb <= choice < unipb + shrpb:
            indx1 = gp.mutShrink(ind)
            ind = indx1[0]
        elif unipb + shrpb <= choice < unipb + shrpb + inspb:
            indx1 = gp.mutInsert(ind, pset=pset)
            ind = indx1[0]
        else:
            choice2 = random.random()
            if choice2 < 0.5:
                indx1 = gp.mutEphemeral(ind, "all")
                ind = indx1[0]
            else:
                indx1 = gp.mutEphemeral(ind, "one")
                ind = indx1[0]
    return ind,

############################ MODIFIED STATISTICS FUNCTIONS  ########################################

def Min(pop):
    """
    The old Min function from the DEAP library was returning incorrect data in case of multiobjective fitness function.
    The stats weren't about one individual, but it was printing the minimum value found for each objective separately,
    also if they didn't belong to the same individual.
    """
    min = pop[0]
    w = np.array([-1.0, -1.0])
    for ind in pop:
        if ind[-1] == 0:
            if min[-1] == 0 and sum(ind[0:2] * w) > sum(min[0:2] * w):
                min = ind
            elif min[-1] != 0:
                min = ind
        elif ind[-1] < min[-1]:
            min = ind
    return min



