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


References:
[1] https://pastebin.com/QKMhafRq
"""

import numpy as np
from functools import partial
import random
from deap import tools
from operator import attrgetter


def InclusiveTournament(mu, organized_pop, good_indexes, selected_individuals, fitness_size, parsimony_size, creator):
    """
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    Rationale behind InclusiveTournament: a double tournament selection is performed in each category, so to maintain
    diversity. Double Tournament is used so to avoid bloat. An exploited measure is used to point out when a category is
    completely exploited. For examples, if in a category are present only 4 individuals, the tournament will be
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


def selDoubleTournament_IGP(individuals, k, fitness_size, parsimony_size, creator, fitness_first):
    """This is a modification of the xselDoubleTournament function from [1] which itself is a modification of
     the selDoubleTournament function implemented in the DEAP library (https://github.com/DEAP/deap/blob/master/deap/tools/selection.py#L106).

     The modification is done in order to deal with individual composed by multiple trees and with the penalty fitness used by the IGP.

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

def selBest_IGP(individuals):
    """
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    This function selects the best individuals in a population composed by individuals with multiple trees and whose
    fitness array is composed by at least two elements. The first element is the main fitness function driving the
    evolutionary process. The second fitness is a penalty term used if constraints are applied to the problem.

    The logic behind this function is to always select feasible individuals (last fitness term = 0). If both individuals
    are feasible then select the best one (either with bigger or smaller fitness according to the problem). If both
    individuals are unfeasible, then 10% of the times, select the one with the lowest penalty, while 90% ot the times,
    select the one with the best sum of all penalties.

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

def InclusiveTournament3D(mu, organized_pop, good_indexes, selected_individuals, fitness_size, parsimony_size, creator):
    """
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    This function is a modification of the Inclusive Tournament. It consists in applying the inclusive tournament
    on a 3D niches space.
    """

    chosen = []
    exploited = np.zeros((len(good_indexes)))
    j = 0
    count = 0
    while len(chosen) < mu:
        if j > len(good_indexes) - 1:
            j = 0
        i = good_indexes[j]

        if exploited[j] < len(organized_pop["cat{}_{}_{}".format(i[0], i[1], i[2])]):
            if len(organized_pop["cat{}_{}_{}".format(i[0], i[1], i[2])]) > 1:
                selected = selDoubleTournament_IGP(organized_pop["cat{}_{}_{}".format(i[0], i[1], i[2])], selected_individuals, fitness_size,
                                               parsimony_size, creator, fitness_first=True)
                chosen.append(selected)
            else:
                chosen.append(organized_pop["cat{}_{}_{}".format(i[0], i[1], i[2])][0])
            exploited[j] += 1
        j += 1
        #if count > 1e3:
        #    print('stuck infinte loop selection')
        count+=1
    return chosen


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