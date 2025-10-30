import numpy as np


def evaluate_regression(individual, compile, **kwargs):
    X_train = kwargs['kwargs']['X_train']
    y_train = kwargs['kwargs']['y_train']
    X_val = kwargs['kwargs']['X_val']
    y_val = kwargs['kwargs']['y_val']

    f_ind = compile(individual)
    pred_train = f_ind(*X_train)
    pred_val = f_ind(*X_val)
    if not hasattr(pred_train, "__len__"):
        pred_train = np.ones(len(y_train)) * pred_train
        pred_val = np.ones(len(y_val)) * pred_val
    err_train = y_train - pred_train
    err_val = y_val - pred_val

    fitness_train = np.sqrt(sum(err_train**2)/(len(err_train)))
    fitness_val = np.sqrt(sum(err_val ** 2) / (len(err_val)))

    return fitness_train, 0.0, fitness_val, 0.0


def evaluate_regression_noVal(individual, compile, **kwargs):
    X_train = kwargs['kwargs']['X_train']
    y_train = kwargs['kwargs']['y_train']

    f_ind = compile(individual)
    pred_train = f_ind(*X_train)
    if not hasattr(pred_train, "__len__"):
        pred_train = np.ones(len(y_train)) * pred_train
    err_train = y_train - pred_train

    fitness_train = np.sqrt(sum(err_train**2)/(len(err_train)))

    return fitness_train, 0.0



