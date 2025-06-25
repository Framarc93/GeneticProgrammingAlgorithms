from deap import gp
import numpy as np
from src.MGGP_utils import build_funcString

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


def evaluate_MGGP(d, individual, compile, input_train, output_train, input_val, output_val):

    eq = build_funcString(d, individual)
    f_ind = compile(eq)
    out_train = f_ind(*input_train)
    out_val = f_ind(*input_val)
    if not hasattr(out_train, "__len__"):
        out_train = np.ones(len(output_train)) * out_train
        out_val = np.ones(len(output_val)) * out_val
    err_train = output_train - out_train
    err_val = output_val - out_val
    return err_train, err_val

