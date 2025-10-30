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
File containing the mutation functions

References:
[1] https://pastebin.com/QKMhafRq
"""

import random
from deap import gp


def xmutMultiple(ind, expr, unipb, shrpb, inspb, pset, creator):
    """
    From [1] and modified. Added several mutations possibilities.
    """
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

def xmut_MGGP_multiple(ind, expr, pset, unipb, nodepb):
    for i in range(len(ind)):
        ch = random.randint(0, len(ind[i])-1)
        choice = random.random()
        if choice < unipb:
            indx1 = gp.mutUniform(ind[i][ch], expr=expr, pset=pset)
            ind[i][ch] = indx1[0]
        elif choice < unipb + nodepb:
            indx1 = gp.mutNodeReplacement(ind[i][ch], pset=pset)
            ind[i][ch] = indx1[0]
        else:
            indx1 = gp.mutEphemeral(ind[i][ch], mode='one')
            ind[i][ch] = indx1[0]
    return ind,