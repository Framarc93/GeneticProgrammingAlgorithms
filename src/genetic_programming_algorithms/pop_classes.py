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

import numpy as np
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from genetic_programming_algorithms.niches_manipulation import subset_diversity_genotype, subset_diversity_pheno3D_2fit


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

    def __init__(self, population, creator, **kwargs):
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


class POP_pheno_3D_2fit(object):
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

    def __init__(self, population, creator, **kwargs):

        if "kwargs" in kwargs.keys():
            kwargs = kwargs["kwargs"]

        self.items = list()
        self.lens = np.zeros(len(population))
        self.fits = np.zeros(len(population))
        self.fits_val = np.zeros(len(population))
        self.entropy = 0
        for i in range(len(population)):
            item = deepcopy(population[i])
            self.items.append(item)
            self.lens[i] = len(population[i])
            self.fits[i] = population[i].fitness.values[0]
            self.fits_val[i] = population[i].fitness_validation.values[0]
        self.min_len = int(min(self.lens))
        self.max_len = int(max(self.lens))

        self.min_fit = min(self.fits)
        self.max_fit = max(self.fits)

        self.min_fit_val = min(self.fits_val)
        self.max_fit_val = max(self.fits_val)

        self.categories, self.indexes = subset_diversity_pheno3D_2fit(population, creator, kwargs=kwargs)
        pp_fit = self.categories["distribution"]["percentage"]
        pp_fit = pp_fit[pp_fit != 0]
        self.entropy = -sum(pp_fit * np.log(pp_fit))

    def output_stats(self, Title, cat_number_len, cat_number_fit):
        print("\n")
        print("--------------------------- {} STATISTICS --------------------------------".format(Title))
        print("-- Min fit: {}, Max fit: {}, Min len: {}, Max len: {}, Min Fit Val: {}, Max Fit Val: {}---".format(self.min_fit,
                                                                                                                self.max_fit, self.min_len, self.max_len,
                                                                                                                self.min_fit_val, self.max_fit_val))
        print("-- Entropy: {0:.3f} -----------------------------".format(self.entropy))
        #print("-- Distribution : --")
        #print(tabulate(np.reshape(self.categories["distribution"]["percentage"] * 100, (cat_number_fit, cat_number_len))))

    def save_stats(self, data, path, gen):
        data = np.vstack((data, [self.min_fit, self.max_fit, self.min_len, self.max_len, self.entropy,
                                 self.categories]))
        np.save(path + 'Population_statistics_gen_{}'.format(gen), data)
        np.save(path + 'Individuals_lengths_gen_{}'.format(gen), self.lens)

    def retrieve_stats(self, data, lengths):
        data = np.vstack((data, np.array(([self.min_len, self.max_len, self.entropy,
                                           self.categories["distribution"]["percentage"]]), dtype=object)))
        lengths.append(self.lens)
        return data, lengths

    def plot_entropy_evol(self, save_path, gen, cat_len, cat_fit, cat_fit_val, fit_scale):
        #plt.figure(figsize=(5, 5))
        #len_range = np.linspace(self.min_len, self.max_len, cat_len+1)
        #fit_range = np.geomspace(self.min_fit, self.max_fit*fit_scale, cat_fit+1)
        #plt.vlines(x=len_range, ymin=self.min_fit, ymax=self.max_fit*fit_scale, colors='k')
        #plt.hlines(y=fit_range, xmin=self.min_len, xmax=self.max_len, colors='k')
        #plt.title('Gen {}'.format(gen))
        #plt.xlabel('Number of nodes')
        #plt.ylabel('Fitness')
        #count = 0
        #for i in range(len(fit_range)-1):
        #    for j in range(len(len_range)-1):
        #        plt.plot((len_range[j]+len_range[j+1])/2, (fit_range[i]+fit_range[i+1])/2, 'o', markersize=self.categories['distribution']['percentage'][count]*100, color='r')
        #        count += 1

        ax = plt.figure().add_subplot(projection='3d')
        plt.rcParams["figure.figsize"] = [6, 6]

        ax.azim = -134  # y rotation (default=270)
        ax.elev = 38  # x rotation (default=0)

        q3 = np.quantile(self.fits, fit_scale)
        q3_val = np.quantile(self.fits_val, fit_scale)
        len_range = np.linspace(self.min_len, self.max_len, cat_len + 1)
        fit_range = np.geomspace(self.min_fit, q3, cat_fit + 1)
        fit_val_range = np.geomspace(self.min_fit_val, q3_val, cat_fit_val + 1)

        for fit_val in fit_val_range:
            for fit in fit_range:
                ax.plot([fit_val, fit_val], [self.min_len, self.max_len], [fit, fit], color='k', linewidth=0.3, alpha=0.3)
        for fit_val in fit_val_range:
            for Len in len_range:
                ax.plot([fit_val, fit_val], [Len, Len], [self.min_fit, q3], color='k', linewidth=0.3, alpha=0.3)
        for fit in fit_range:
            for Len in len_range:
                ax.plot( [self.min_fit_val, q3_val], [Len, Len], [fit, fit], color='k', linewidth=0.3, alpha=0.3)
        count = 0
        for i in range(len(fit_range)-1):
            for j in range(len(len_range)-1):
                for k in range(len(fit_val_range)-1):
                    ax.plot((fit_val_range[k]+fit_val_range[k+1])/2, (len_range[j]+len_range[j+1])/2, (fit_range[i]+fit_range[i+1])/2, 'o', markersize=self.categories['distribution']['percentage'][count]*150, color='r')
                    count += 1
        plt.title('Gen {}'.format(gen))
        ax.set_xlabel('Validation Fitness')
        ax.set_ylabel('Length')
        ax.set_zlabel('Training Fitness')

        #plt.show()
        plt.savefig(save_path + '/gen_{}.jpg'.format(gen), format='jpg')
        plt.close()

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