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


from deap import gp, creator, base, tools
import operator
import numpy as np
import common.GP_PrimitiveSet as gpprim
import random
from FIGP.FIGP_functions import InclusiveTournament3D
from examples.regression.evaluate_functions import evaluate_IGP_FIGP
from IGP.IGP_functions import xmutMultiple
from functools import partial

def define_FIGP_model(terminals, nEph, Eph_max, limit_height, limit_size, n):
    ####################################    P R I M I T I V E  -  S E T     ################################################

    pset = gp.PrimitiveSet("Main", terminals)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(gpprim.TriAdd, 3)
    pset.addPrimitive(gpprim.TriMul, 3)
    pset.addPrimitive(np.tanh, 1)
    pset.addPrimitive(gpprim.Square, 1)
    pset.addPrimitive(gpprim.ModLog, 1)
    pset.addPrimitive(gpprim.ModExp, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)

    def ephemeral_creation(Eph_max):
        return round(random.uniform(-Eph_max, Eph_max), 4)

    for i in range(nEph):
        pset.addEphemeralConstant("rand{}_{}".format(i, n), partial(ephemeral_creation, Eph_max=Eph_max))

    #for j in range(nEph):
    #    pset.addEphemeralConstant("rand_demes_{}_{}".format(j, n_iter), lambda: round(random.uniform(-12, 12), 4))

    for i in range(len(pset.arguments)):
        pset.arguments[i] = 'x{}'.format(i)
        pset.mapping['x{}'.format(i)] = pset.mapping['ARG{}'.format(i)]
        pset.mapping['x{}'.format(i)].value = 'x{}'.format(i)
        del pset.mapping['ARG{}'.format(i)]

    ################################################## TOOLBOX #############################################################

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, fitness_validation=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate_IGP_FIGP)
    toolbox.register("select", InclusiveTournament3D, selected_individuals=1, fitness_size=2, parsimony_size=1.6, creator=creator)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)
    toolbox.register("mutate", xmutMultiple, expr=toolbox.expr_mut, unipb=0.55, shrpb=0.05, inspb=0.25, pset=pset,
                     creator=creator)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

    return pset, creator, toolbox