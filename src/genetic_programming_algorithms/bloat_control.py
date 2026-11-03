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

import random
from copy import deepcopy
from functools import wraps

from deap import gp


def staticLimit_subTrees(key, max_value):
    """Replace every oversized subtree with a copy of a parent subtree."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            parents = deepcopy(args)
            children = list(func(*args, **kwargs))
            for child in children:
                for j, tree in enumerate(child):
                    if key(tree) > max_value:
                        candidates = [
                            parent[j]
                            for parent in parents
                            if j < len(parent) and key(parent[j]) <= max_value
                        ]
                        if not candidates:
                            raise ValueError("No parent subtree satisfies the size limit")
                        child[j] = deepcopy(random.choice(candidates))
            return children

        return wrapper

    return decorator


def staticLimit_mutShrink(key, max_value):
    """Shrink oversized genes; reject impossible limits instead of looping forever."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            children = list(func(*args, **kwargs))

            def shrink(tree):
                if not isinstance(tree, gp.PrimitiveTree):
                    for gene in tree:
                        shrink(gene)
                    return
                while key(tree) > max_value:
                    before = len(tree)
                    gp.mutShrink(tree)
                    if len(tree) == before:
                        raise ValueError("Tree cannot be shrunk to the requested limit")

            for child in children:
                for tree in child:
                    shrink(tree)
            return children

        return wrapper

    return decorator
