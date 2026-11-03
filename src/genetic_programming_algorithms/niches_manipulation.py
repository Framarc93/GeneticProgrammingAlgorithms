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


def subset_feasible(population):
    """
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    Function used to create a subset of feasible individuals from the input population"""
    sub_pop = []
    for ind in population:
        if ind.fitness.values[-1] == 0:
            sub_pop.append(ind)
    return sub_pop


def subset_diversity_genotype(population, creator, **kwargs):
    """
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    Function used to divide the individuals in the input population into cat_number categories. The division into
    categories is done according to the length of the individuals in the population
    """
    cat_number = 10  # here the number of categories is selected
    lens = []
    categories = {}
    distribution = []
    distr_stats = {}
    invalid_ind = []

    for ind in population:
        try:
            if isinstance(ind[0], creator.SubIndividual) and len(ind) > 1:
                lens.append(sum([len(gpt) for gpt in ind]))
            else:
                lens.append(len(ind))
        except AttributeError:
            lens.append(len(ind))
    cat = np.linspace(min(lens), max(lens), cat_number + 1)
    useful_ind = np.linspace(0, len(cat) - 2, len(cat) - 1)

    for i in range(len(cat) - 1):
        categories[f"cat{i}"] = []
    for ind in population:
        for i in range(len(cat) - 1):
            try:
                if isinstance(ind[0], creator.SubIndividual):
                    totLen = sum([len(gpt) for gpt in ind])
                else:
                    totLen = len(ind)
            except AttributeError:
                totLen = len(ind)
            if cat[i] <= totLen <= cat[i + 1]:
                categories[f"cat{i}"].append(ind)
                break

    for i in range(len(cat) - 1):
        if not categories[f"cat{i}"]:
            invalid_ind.append(i)
        distribution.append(len(categories[f"cat{i}"]))
    distr_stats["individuals"] = distribution
    distr_stats["percentage"] = np.array(distribution) / len(population)
    categories["distribution"] = distr_stats
    if invalid_ind:
        useful_ind = np.delete(useful_ind, invalid_ind, 0)
    return categories, np.asarray(useful_ind, dtype=int)


def subset_diversity_pheno3D_2fit(population, creator, **kwargs):
    """
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    Function used to divide the individuals in the input population into cat_number categories. The division into
    categories is done according to the training fitness, validation fitness and length of the individuals
    in the population
    """

    cat_number_len = kwargs["kwargs"]["configs"]["cat_number_len"]
    cat_number_fit_train = kwargs["kwargs"]["configs"]["cat_number_fit_train"]
    cat_number_fit_val = kwargs["kwargs"]["configs"]["cat_number_fit_val"]
    fit_scale = kwargs["kwargs"]["configs"]["fit_scale"]

    fits = []
    categories = {}
    distribution = []
    distr_stats = {}
    useful_ind = []
    lens = []
    fits_val = []
    for ind in population:
        fits.append(ind.fitness.values[0])
        fits_val.append(ind.fitness_validation.values[0])
        lens.append(len(ind))

    fits = np.array(fits)
    fits_val = np.array(fits_val)
    lens = np.array(lens)

    upper_fit = np.quantile(fits, fit_scale)
    upper_fit_val = np.quantile(fits_val, fit_scale)

    int_lens = np.linspace(min(lens), max(lens), cat_number_len + 1)
    int_fits = (np.geomspace if min(fits) > 0 else np.linspace)(
        min(fits), upper_fit, cat_number_fit_train + 1
    )
    int_fits_val = (np.geomspace if min(fits_val) > 0 else np.linspace)(
        min(fits_val), upper_fit_val, cat_number_fit_val + 1
    )

    for i in range(cat_number_fit_train):
        for j in range(cat_number_len):
            for k in range(cat_number_fit_val):
                categories[f"cat{i}_{j}_{k}"] = []

    for ind in population:
        break_out_flag = False
        fit = ind.fitness.values[0]
        fit_val = ind.fitness_validation.values[0]
        ind_len = len(ind)
        for i in range(cat_number_fit_train):
            if i == cat_number_fit_train - 1:
                for j in range(cat_number_len):
                    if int_lens[j] <= ind_len <= int_lens[j + 1]:
                        for k in range(cat_number_fit_val):
                            if k == cat_number_fit_val - 1:
                                categories[f"cat{i}_{j}_{k}"].append(ind)
                                break_out_flag = True
                                break
                            elif int_fits_val[k] <= fit_val <= int_fits_val[k + 1]:
                                categories[f"cat{i}_{j}_{k}"].append(ind)
                                break_out_flag = True
                                break
                        if break_out_flag is True:
                            break
                if break_out_flag is True:
                    break
            elif int_fits[i] <= fit <= int_fits[i + 1]:
                for j in range(cat_number_len):
                    if int_lens[j] <= ind_len <= int_lens[j + 1]:
                        for k in range(cat_number_fit_val):
                            if k == cat_number_fit_val - 1:
                                categories[f"cat{i}_{j}_{k}"].append(ind)
                                break_out_flag = True
                                break
                            elif int_fits_val[k] <= fit_val <= int_fits_val[k + 1]:
                                categories[f"cat{i}_{j}_{k}"].append(ind)
                                break_out_flag = True
                                break
                        if break_out_flag is True:
                            break
                if break_out_flag is True:
                    break

    for i in range(cat_number_fit_train):
        for j in range(cat_number_len):
            for k in range(cat_number_fit_val):
                if categories[f"cat{i}_{j}_{k}"]:
                    useful_ind.append([i, j, k])
                distribution.append(len(categories[f"cat{i}_{j}_{k}"]))

    distr_stats["individuals"] = distribution
    distr_stats["percentage"] = np.array(distribution) / len(population)
    categories["distribution"] = distr_stats

    return categories, np.asarray(useful_ind, dtype=int)
