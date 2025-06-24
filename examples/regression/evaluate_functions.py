from deap import gp
import numpy as np

def evaluate_IGP_FIGP(individual, pset, **kwargs):
    X_train = kwargs['kwargs']['X_train']
    y_train = kwargs['kwargs']['y_train']
    X_val = kwargs['kwargs']['X_val']
    y_val = kwargs['kwargs']['y_val']
    f_ind = gp.compile(individual, pset=pset)
    out_train = f_ind(*X_train)
    out_val = f_ind(*X_val)
    if not hasattr(out_train, '__len__'):
        out_train = out_train * np.ones(len(y_train))
        out_val = out_val * np.ones(len(y_val))
    err_train = y_train - out_train
    err_val = y_val - out_val
    fitness_train = np.sqrt(sum(err_train**2)/(len(err_train)))
    fitness_val = np.sqrt(sum(err_val ** 2) / (len(err_val)))
    return fitness_train, 0.0, fitness_val, 0.0

