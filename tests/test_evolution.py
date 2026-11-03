import random

import numpy as np
import pytest
from deap import creator

from examples.regression.evaluate_functions import evaluate_regression
from genetic_programming_algorithms import MGGP_utils as m
from genetic_programming_algorithms import gp_model_definition_functions as models
from genetic_programming_algorithms.main_functions import main_evolProcess


@pytest.mark.parametrize("kind", ["IGP", "FIGP", "MGGP"])
def test_evolution(kind):
    random.seed(123)
    for name in ["Fitness", "Individual", "SubIndividual", "Gene"]:
        if hasattr(creator, name):
            delattr(creator, name)
    x = np.linspace(0.1, 1, 8)
    settings = {
        "fitness_validation": True,
        "NgenesMax": 3,
        "stdCxpb": 0.5,
        "evaluate_regression": evaluate_regression,
    }
    evaluate = m.lst_matrix if kind == "MGGP" else evaluate_regression
    pset, cr, tb = getattr(models, "define_" + kind + "_model")(
        1, 1, 1, 1, 10, 100, 0, evaluate, ["add", "mul"], kwargs=settings
    )
    configs = {
        "init_repeat": 1,
        "save_gen": False,
        "save_pop": False,
        "fit_tol": None,
        "cx_lim": 0.8,
        "cat_number_len": 2,
        "cat_number_fit_train": 2,
        "cat_number_fit_val": 2,
        "fit_scale": 0.5,
    }
    result = main_evolProcess(
        12,
        3,
        12,
        6,
        0.5,
        0.5,
        1,
        pset,
        cr,
        tb,
        configs=configs,
        X_train=x[None, :],
        y_train=x + 2,
        X_val=x[None, :],
        y_val=x + 2,
    )
    pop, log, hof, stats, lengths, _ = result
    assert len(pop) == 12 and len(log) == 3 and len(lengths) == 3
    assert all(ind.fitness.valid for ind in pop)
    assert len(hof) > 0


@pytest.mark.parametrize("validation", [False, True])
def test_multi_output_fit(validation):
    class Individual(list):
        pass

    x = np.arange(5.0)
    ind = [Individual(["x"]), Individual(["x"])]
    fn = m.lst_matrix_multiple_subInds if validation else m.lst_matrix_multiple_subInds_noVal
    result = fn(
        ind,
        lambda *a: x,
        lambda *a, **kw: (0, 0, 0, 0),
        None,
        kwargs={"X_train": x[None, :], "y_train": np.array([2 + 3 * x, 4 + 5 * x])},
    )
    np.testing.assert_allclose(result[-2], [2, 3], atol=1e-12)
    np.testing.assert_allclose(result[-1], [4, 5], atol=1e-12)
