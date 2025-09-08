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

from deap import gp
from copy import deepcopy, copy
from functools import wraps
import random


def staticLimit_subTrees(key, max_value):
    """
    This is a modification of the staticLimit function implemented in the DEAP library
    (https://github.com/DEAP/deap/blob/master/deap/gp.py#L908) in order to deal with individual composed by multiple
    trees.
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


def staticLimit_mutShrink(key, max_value):
    """
    This is a modification of the staticLimit function implemented in the DEAP library
    (https://github.com/DEAP/deap/blob/master/deap/gp.py#L908). The modification is done in order to deal with
    individuals composed by multiple trees used by the MGGP and the mutShrink operator is applied to reduce the size
    of an individual
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