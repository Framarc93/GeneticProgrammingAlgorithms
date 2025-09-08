# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.

# ------------ Copyright (C) 2025 Francesco Marchetti -----------------
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
This file contains the function used to retrieve the data for the regression application
"""


from pmlb import fetch_data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# Base folder: where the script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def retrieve_dataset(bench):
    '''
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    This function download the selected dataset from the pmlb repository https://epistasislab.github.io/pmlb/

    Attributes:
        bench: string
            Name of the dataset to download
    '''

    if os.path.exists(BASE_DIR + "/{}_shuffled.csv".format(bench)):
        # if the file of the shuffled dataset already exist, then do nothing.
        return
    else:
        # retrieve the dataset
        data = fetch_data(bench)

        # shuffle the dataset
        shuffled = data.sample(frac=1)

        # save the dataset to csv
        shuffled.to_csv(BASE_DIR + '/{}_shuffled.csv'.format(bench), index=False)
        return


def select_testcase(bench, test_perc, val_perc):
    '''
    Author(s): Francesco Marchetti
    email: framarc93@gmail.com

    This function retrieves the shuffled dataset, performs a MinMax scaling over it and split it into train,
    validation and test

    Attributes:
        bench: string
            Name of the dataset to download
        test_perc: float
            percentage of data to use for test
        val_perc: float
            percentage of data to use for validation
    '''

    df = pd.read_csv(BASE_DIR + '/{}_shuffled.csv'.format(bench))
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    input = data.iloc[:, :-1]
    target = data.iloc[:, -1]
    terminals = len(data.columns) - 1
    input_train, X_test, target_train, y_test = train_test_split(input, target, test_size=test_perc)

    if val_perc == 0.0:
        X_train, y_train = input_train, target_train
        X_val, y_val = X_test, y_test
    else:
        # Use the same function above for the validation set
        X_train, X_val, y_train, y_val = train_test_split(input_train, target_train, test_size=val_perc)

    print("X_train shape: {}".format(X_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("X_test shape: {}".format(X_test.shape))
    print("y_test shape: {}".format(y_test.shape))
    print("X_val shape: {}".format(X_val.shape))
    print("y val shape: {}".format(y_val.shape))

    return (terminals, X_train.values.T, y_train.values.T, X_val.values.T,
            y_val.values.T, X_test.values.T, y_test.values.T)
