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
This script is the main for the control application. The user must select the algorithm and the test case
"""

from src.main_functions import main_evolProcess
import yaml
from yaml.loader import SafeLoader
import os
import numpy as np
from time import time
import operator
from src.gp_model_definition_functions import define_IGP_model
import sympy
from benchmarks.inverted_pendulum.evaluate import evaluate_pendulum
import benchmarks.inverted_pendulum.Plant as PlantPendulum
import multiprocess
from propagation_functions import propagate_forward
from benchmarks.inverted_pendulum import dynamics
#################################################################################################################

# This section must remain outside the if __name__=="__main__" otherwise there are issue with the multiprocessing

#################################################################################################################

algo  = "IGP"         # select the GP algorithm. Choose between IGP, FIGP and MGGP
bench = "pendulum"    # select the benchmark

match algo:
    case "IGP":
        define_GP_model = define_IGP_model
    case "FIGP":
        ##### Currently only the IGP is configured to work for control applications
        pass
    case "MGGP":
        ##### Currently only the IGP is configured to work for control applications
        pass
    case _:
        print("Select a GP algorithm between IGP, FIGP and MGGP.")

if bench == "pendulum":
    plant = PlantPendulum.Pendulum()
    evaluation_function = evaluate_pendulum
    dynamics = dynamics.dynamics_pendulum

with open('control_config.yaml') as f:
    configs = yaml.load(f, Loader=SafeLoader) # load configs

nEph = configs['nEph']
Eph_max = configs['Eph_max']
limit_height = configs['limit_height']  # Max height (complexity) of the gp law
limit_size = configs['limit_size']  # Max size (complexity) of the gp law
size_pop = configs['size_pop']
size_gen = configs['size_gen']  # Gen size
fit_tol = configs['fit_tol']
ntot = 1  # configs['ntot']
save_gen = configs['save_gen']
save_pop = configs['save_pop']
mutpb = configs['mutpb']
cxpb = configs['cxpb']
cx_lim = configs['cx_lim']

Mu = int(size_pop)
Lambda = int(size_pop * 1.2)
nbCPU = 1#multiprocess.cpu_count()  # threads to use

# create save folder
save_path = configs["save_path"] + '{}_{}/'.format(algo, bench)
try:
    os.makedirs(save_path)
except FileExistsError:
    pass

# define quantities to save
to_save = np.array(['t_evaluate', 'RMSE_train', 'RMSE_test'])

for n in range(ntot):
    pset, creator, toolbox = define_GP_model(plant.n_states, plant.n_controls, nEph, Eph_max, limit_height, limit_size, n, evaluation_function, fitness_validation=False)

    if __name__ == "__main__":

        save_path_iter = save_path + 'Sim{}/'.format(n)
        try:
            os.makedirs(save_path_iter)
        except FileExistsError:
            pass
        print("----------------------- Iteration {} ---------------------------".format(n))

        start = time()
        pop, log, hof, pop_statistics, ind_lengths, pset = main_evolProcess(size_pop, size_gen, Mu, Lambda, cxpb, mutpb,
                                                                            nbCPU, pset, creator, toolbox,
                                                                            configs=configs, save_path_iter=save_path_iter,
                                                                            plant=plant, dynamics=dynamics)


        end = time()

        t_offdesign = end - start

        best_ind = hof[-1]
        best_ind = hof.items[-1]

        X, V, Theta, Omega = sympy.symbols('eX eV eTheta eOmega')
        sympy.init_printing(use_unicode=True)

        simplified_eq = sympy.sympify(str(best_ind),
                                      locals={'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 'cos': sympy.cos,
                                              'sin': sympy.sin})

        with open(save_path_iter + 'best_ind_structure_IGP.txt', 'w') as f:
            f.write(str(simplified_eq))

        t_offdesign = end - start

        np.save(save_path_iter + 'computational_time_IGP.npy', t_offdesign)
        best_ind = hof[-1]
        x_IGP, u_IGP, failure = propagate_forward(plant, best_ind, toolbox.compile, dynamics)
        np.save(save_path_iter + 'x_IGP.npy', x_IGP)
        np.save(save_path_iter + 'u_IGP.npy', u_IGP)

