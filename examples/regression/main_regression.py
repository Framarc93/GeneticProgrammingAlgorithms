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
This script is the main for the regression application. The user must select the algorithm and the benchmark
"""

from IGP.main_functions import main_IGP_regression
from FIGP.main_functions import main_FIGP_regression
from MGGP.main_functions import main_MGGP_regression
from data.data_handling import retrieve_dataset
import yaml
from yaml.loader import SafeLoader
import os
import numpy as np
from examples.regression.data.data_handling import select_testcase
from time import time
from deap import gp
from copy import copy
import matplotlib.pyplot as plt
from src.gp_model_definition_functions import define_IGP_model, define_FIGP_model, define_MGGP_model
from src.MGGP_utils import build_funcString
from evaluate_functions import evaluate_MGGP, evaluate_IGP_FIGP

#################################################################################################################

# This section must remain outside the if __name__=="__main__" otherwise there are issue with the multiprocessing

#################################################################################################################

algo  = "MGGP"         # select the GP algorithm. Choose between IGP, FIGP and MGGP
bench = "503_wind"    # select the benchmark

match algo:
    case "IGP":
        main_function = main_IGP_regression
        define_GP_model = define_IGP_model
        evaluation_function = evaluate_IGP_FIGP
    case "FIGP":
        main_function = main_FIGP_regression
        define_GP_model = define_FIGP_model
        evaluation_function = evaluate_IGP_FIGP
    case "MGGP":
        main_function = main_MGGP_regression
        define_GP_model = define_MGGP_model
        evaluation_function = evaluate_MGGP
    case _:
        print("Select a GP algorithm between IGP, FIGP and MGGP.")


retrieve_dataset(bench) # download and shuffle dataset

with open('regression_config.yaml') as f:
    configs = yaml.load(f, Loader=SafeLoader) # load configs

nEph = configs['nEph']
Eph_max = configs['Eph_max']
limit_height = configs['limit_height']  # Max height (complexity) of the gp law
limit_size = configs['limit_size']  # Max size (complexity) of the gp law
size_pop = configs['size_pop']
size_gen = configs['size_gen']  # Gen size
cat_number_len = configs['cat_number_len']
cat_number_fit = configs['cat_number_fit']
cat_number_height = configs['cat_number_height']
fit_scale = configs['fit_scale']
fit_tol = configs['fit_tol']
ntot = 1  # configs['ntot']
save_gen = configs['save_gen']
save_pop = configs['save_pop']
mutpb = configs['mutpb']
cxpb = configs['cxpb']
cx_lim = configs['cx_lim']
test_perc = configs['test_perc']
val_perc = configs['val_perc']
NgenesMax = configs['NgenesMax']
stdCxpb = configs['stdCxpb']

Mu = int(size_pop)
Lambda = int(size_pop * 1.2)
nbCPU = 1#multiprocess.cpu_count() # threads to use

# create save folder
save_path = configs["save_path"] + '{}_{}/'.format(algo, bench)
try:
    os.makedirs(save_path)
except FileExistsError:
    pass

# define quantities to save
to_save = np.array(['t_evaluate', 'RMSE_train', 'RMSE_test'])

terminals, X_train, y_train, X_val, y_val, X_test, y_test = select_testcase(bench, test_perc, val_perc)

for n in range(ntot):

    pset, creator, toolbox = define_GP_model(terminals, nEph, Eph_max, limit_height, limit_size, n, evaluation_function,
                                             kwargs={'NgenesMax': NgenesMax, 'stdCxpb': stdCxpb})

    if __name__ == "__main__":

        save_path_iter = save_path + 'Sim{}/'.format(n)
        try:
            os.makedirs(save_path_iter)
        except FileExistsError:
            pass
        print("----------------------- Iteration {} ---------------------------".format(n))

        start = time()

        pop, log, hof, pop_statistics, ind_lengths, pset = main_function(size_pop, size_gen, Mu, Lambda, cxpb, mutpb,
                                                                         nbCPU, X_train, y_train, X_val, y_val,pset,
                                                                         creator, toolbox, save_path_iter, save_pop,
                                                                         save_gen, kwargs={'fit_tol': fit_tol,
                                                                                           'terminals': terminals,
                                                                                           'cx_lim': cx_lim,
                                                                                           'cat_number_len': cat_number_len,
                                                                                           'cat_number_fit': cat_number_fit,
                                                                                           'cat_number_height': cat_number_height,
                                                                                           'fit_scale': fit_scale})


        end = time()

        t_offdesign = end - start

        best_ind = hof[-1]
        if algo == "MGGP":
            string = str(best_ind.w[0])
            st = 1
            while st <= len(best_ind):
                string = string + "+" + str(best_ind.w[st]) + "*" + str(best_ind[st - 1])
                st += 1
            print("\n Best training individual: ", string)
        else:
            print("\n Best training individual: ", best_ind)
        print("Best training fitness:", best_ind.fitness.values[0])

        fitness_test = []
        min_test = 100
        RMSE_train = 0
        for i in range(size_gen):
            best_ind = np.load(save_path_iter + 'Best_ind_{}'.format(i), allow_pickle=True)

            if algo == "MGGP":
                string = str(best_ind.w[0])
                st = 1
                while st <= len(best_ind):
                    string = string + "+" + str(best_ind.w[st]) + "*" + str(best_ind[st - 1])
                    st += 1
                best_ind = build_funcString(best_ind.w, best_ind)

            f = gp.compile(best_ind, pset=pset)
            y_best_test = f(*X_test)
            err_test = y_test - y_best_test
            RMSE_test = np.sqrt(sum(err_test ** 2) / (len(err_test)))
            fitness_test.append(RMSE_test)
            if RMSE_test < min_test:
                min_test = copy(RMSE_test)
                y_best_train = f(*X_train)
                err_train = y_train - y_best_train
                RMSE_train = np.sqrt(sum(err_train ** 2) / (len(err_train)))

        # Plot train, validation and test fitness evolution
        plt.figure(0)
        plt.plot(np.linspace(0, size_gen - 1, size_gen), np.array(log.chapters["fitness"].select("min"))[:,0], label='Train')
        plt.plot(np.linspace(0, size_gen - 1, size_gen), np.array(log.chapters["fitness_val"].select("min"))[:,0], label='Validation')
        plt.plot(np.linspace(0, size_gen - 1, size_gen), fitness_test, label='Test')
        plt.xlabel('Generations')
        plt.ylabel('RMSE')
        plt.legend(loc='best')
        plt.savefig(save_path_iter + "fitness_evol.jpg".format(n))

        # save data
        np.save(save_path + '{}_{}_{}_train_fit_evol.npy'.format(n, algo, bench), np.array(log.chapters["fitness"].select("min"))[:,0])
        np.save(save_path + '{}_{}_{}_test_fit_evol.npy'.format(n, algo, bench), fitness_test)
        np.save(save_path + '{}_{}_{}_val_fit_evol.npy'.format(n, algo, bench), np.array(log.chapters["fitness_val"].select("min"))[:,0])
        np.save(save_path + "{}_{}_{}_POP_STATS".format(n, algo, bench), pop_statistics)
        np.save(save_path + "{}_{}_{}_IND_LENGTHS".format(n, algo, bench), ind_lengths)
        np.save(save_path + "{}_{}_{}_GEN".format(n, algo, bench), log.select("gen"))
        np.save(save_path + "{}_{}_{}_FIT".format(n, algo, bench), np.array(log.chapters['fitness'].select('min')))
        to_save = np.vstack((to_save, [t_offdesign, RMSE_train, min(fitness_test)]))
        np.save(save_path + "{}_{}_{}_T_RMSE".format(n, algo, bench), to_save)