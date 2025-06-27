from deap import gp
import numpy as np
from src.MGGP_utils import build_funcString


def compute_error(individual, compile, X_train, y_train, X_val, y_val):

    f_ind = compile(individual)
    pred_train = f_ind(*X_train)
    pred_val = f_ind(*X_val)
    if not hasattr(pred_train, "__len__"):
        pred_train = np.ones(len(y_train)) * pred_train
        pred_val = np.ones(len(y_val)) * pred_val
    err_train = y_train - pred_train
    err_val = y_val - pred_val

    return err_train, err_val

def evaluate_IGP_FIGP(individual, compile, **kwargs):
    X_train = kwargs['kwargs']['X_train']
    y_train = kwargs['kwargs']['y_train']
    X_val = kwargs['kwargs']['X_val']
    y_val = kwargs['kwargs']['y_val']

    err_train, err_val = compute_error(individual, compile, X_train, y_train, X_val, y_val)

    fitness_train = np.sqrt(sum(err_train**2)/(len(err_train)))
    fitness_val = np.sqrt(sum(err_val ** 2) / (len(err_val)))

    return fitness_train, 0.0, fitness_val, 0.0


def evaluate_MGGP(d, individual, compile, X_train, y_train, X_val, y_val):

    eq = build_funcString(d, individual)

    err_train, err_val = compute_error(eq, compile, X_train, y_train, X_val, y_val)

    return err_train, err_val

