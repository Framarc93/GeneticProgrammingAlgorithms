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

References:
[1] https://pastebin.com/QKMhafRq


"""

import numpy as np
from deap import gp, tools
import random
from copy import deepcopy
from functools import partial, wraps
from operator import attrgetter

def selDoubleTournament_MGGP(individuals, k, fitness_size, parsimony_size, fitness_first, fit_attr="fitness"):
    """
    This is a modification of the xselDoubleTournament function from [1] which itself is a modification of
     the selDoubleTournament function implemented in the DEAP library (https://github.com/DEAP/deap/blob/master/deap/tools/selection.py#L106).

     The modification is done in order to deal with individual composed by multiple trees used by the MGGP.

    """
    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, k, select):
        """
        From [1] and modified to check for SubIndividual attribute

        """
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            prob = parsimony_size / 2.
            ind1, ind2 = select(individuals, k=2)

            lind1 = sum([len(gpt) for gpt in ind1])
            lind2 = sum([len(gpt) for gpt in ind2])
            if lind1 > lind2:
                ind1, ind2 = ind2, ind1
            elif lind1 == lind2:
                # random selection in case of a tie
                prob = 0.5

            # Since size1 <= size2 then ind1 is selected
            # with a probability prob
            chosen.append(ind1 if random.random() < prob else ind2)

        return chosen

    def _fitTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            aspirants = select(individuals, k=fitness_size)
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament, select=tools.selRandom)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, k, tsize)

def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """
    This function is a modification of the VarOr function implemented in the DEAP library (https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L192)

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

def staticLimit_MGGP(key, max_value):
    """This is a modification of the staticLimit function implemented in the DEAP library (https://github.com/DEAP/deap/blob/master/deap/gp.py#L908).

     The modification is done in order to deal with individuals composed by multiple trees used by the MGGP.

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                to_check = []
                for indd in new_inds[i]:
                    to_check.append(indd)
                check = max(to_check, key=key)
                j = 0
                while j<len(to_check)-1:
                    if key(ind[j]) == key(check):
                        break
                    j += 1
                while key(check) > max_value:
                    new_inds[i][j] = gp.mutShrink(new_inds[i][j])[0]
            return new_inds

        return wrapper

    return decorator


def xmut_MGGP(ind, expr, pset, unipb, nodepb):
    ch = random.randint(0, len(ind)-1)
    choice = random.random()
    if choice < unipb:
        indx1 = gp.mutUniform(ind[ch], expr=expr, pset=pset)
        ind[ch] = indx1[0]
    elif choice < unipb + nodepb:
        indx1 = gp.mutNodeReplacement(ind[ch], pset=pset)
        ind[ch] = indx1[0]
    else:
        indx1 = gp.mutEphemeral(ind[ch], mode='one')
        ind[ch] = indx1[0]
    return ind,


def xmate_MGGP(ind1, ind2, NgenesMax, stdCxpb):
    choice = random.random()
    if choice <= stdCxpb or len(ind1) == 1 or len(ind2) == 1:
        ch1 = random.randint(0, len(ind1) - 1)
        ch2 = random.randint(0, len(ind2) - 1)
        ind1[ch1], ind2[ch2] = gp.cxOnePoint(ind1[ch1], ind2[ch2])

    else:
        ch1start = random.randint(0, len(ind1) - 1)
        ch1end = random.randint(ch1start, len(ind1) - 1)
        ch2start = random.randint(0, len(ind2) - 1)
        ch2end = random.randint(ch2start, len(ind2) - 1)
        from_ind1 = deepcopy([ind1[ch1start]])
        if ch1end != ch1start:
            n = ch1start+1
            while n <= ch1end:
                from_ind1.append(deepcopy(ind1[n]))
                n += 1
        from_ind2 = deepcopy([ind2[ch2start]])
        if ch2end != ch2start:
            n = ch2start + 1
            while n <= ch2end:
                from_ind2.append(deepcopy(ind2[n]))
                n += 1
        if ch1start != ch1end:
            for i in range(ch1end-ch1start+1):
                del ind1[ch1start]
        else:
            del ind1[ch1start]
        ind1[ch1start:ch1start] = deepcopy(from_ind2)
        if ch2start != ch2end:
            for i in range(ch2end-ch2start+1):
                del ind2[ch2start]
        else:
            del ind2[ch2start]
        ind2[ch2start:ch2start] = deepcopy(from_ind1)
        while len(ind1) > NgenesMax:
            del ind1[random.randint(0, len(ind1)-1)]
        while len(ind2) > NgenesMax:
            del ind2[random.randint(0, len(ind2)-1)]

    return ind1, ind2


def initRepeatRandom(container, func, n):
    """
    This is a modification of the initRepeat function implemented in the DEAP library (https://github.com/DEAP/deap/blob/master/deap/tools/init.py#L1)

    """

    return container(func() for _ in range(random.randint(1, n)))