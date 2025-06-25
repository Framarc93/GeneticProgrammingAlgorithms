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
[2] https://sites.google.com/site/gptips4matlab/
"""


from deap import gp
import random
from copy import deepcopy


def xmateMultiple(ind1, ind2):
    """
    From [1] and modified. This function is used to perform crossover in individuals composed by multiple subtrees.
    """
    for i in range(len(ind1)):
        ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2


def xmate_MGGP(ind1, ind2, NgenesMax, stdCxpb):
    """

    This function applied crossover between two MGGP individuals, following the logic implemented in GPTIPS [2]

    Args:
        ind1: first parent individual
        ind2: second parent individual
        NgenesMax: maximum number of genes in an individual
        stdCxpb: probability of applying standard crossover

    Returns:
        ind1: first offspring individual
        ind2: second offspring individual

    """
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