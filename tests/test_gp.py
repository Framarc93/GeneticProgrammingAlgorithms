import random
from types import SimpleNamespace

import numpy as np
import pytest
from deap import base, gp

from examples.control.propagation_functions import RK4
from examples.regression.evaluate_functions import evaluate_regression, evaluate_regression_noVal
from genetic_programming_algorithms import MGGP_utils as m
from genetic_programming_algorithms import crossover_functions as cx
from genetic_programming_algorithms import gp_model_definition_functions as models
from genetic_programming_algorithms import mutation_functions as mut
from genetic_programming_algorithms import primitive_set as p
from genetic_programming_algorithms.pop_init import pop_init_pheno_geno
from genetic_programming_algorithms.utils import HallOfFame_modified, Min, initRepeatRandom


@pytest.fixture(autouse=True)
def seed():
    random.seed(42)
    np.random.seed(42)


@pytest.mark.parametrize(
    "name,args,expected",
    [
        ("lf", (0,), 0.5),
        ("posTanh", (0,), 0),
        ("posSub", (1, 3), 2),
        ("posTriAdd", (-1, -2, 1), 2),
        ("TriAdd", (1, 2, 3), 6),
        ("TriMul", (2, 3, 4), 24),
        ("posAdd", (-3, 1), 2),
        ("posMul", (-3, 2), 6),
        ("Identity", (2,), 2),
        ("Neg", (2,), -2),
        ("Abs", (-2,), 2),
        ("Div", (1, 0), 0),
        ("Mul", (2, 3), 6),
        ("ModSqrt", (-1,), 0),
        ("ModLog", (1,), 0),
        ("ModExp", (0,), 1),
        ("Square", (-2,), 4),
        ("tanh_der", (0,), 1),
    ],
)
def test_primitives(name, args, expected):
    assert getattr(p, name)(*args) == pytest.approx(expected)


@pytest.mark.parametrize("fn", [p.ModLog, p.ModExp, p.Square])
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_inputs_not_modified(fn, dtype):
    x = np.array([-1000, 0, 1000], dtype=dtype)
    original = x.copy()
    assert np.isfinite(fn(x)).all()
    np.testing.assert_array_equal(x, original)
    assert np.isfinite(fn(dtype(2)))


def test_registration():
    names = [
        "lf",
        "posTanh",
        "posSub",
        "posTriAdd",
        "TriAdd",
        "TriMul",
        "posAdd",
        "posMul",
        "Identity",
        "Neg",
        "Abs",
        "Div",
        "Mul",
        "ModSqrt",
        "ModLog",
        "ModExp",
        "Square",
        "tanh_der",
        "add",
        "sub",
        "mul",
        "tanh",
        "sin",
        "cos",
    ]
    pset = p.create_primitive_set(gp.PrimitiveSet("test", 1), names)
    assert set(names) <= pset.mapping.keys()


def test_empty_hall_of_fame():
    hof = HallOfFame_modified(0)
    hof.update([object()], True)
    assert len(hof) == 0


def test_rk4():
    result = RK4(0, 1, lambda t, x: x, 101, np.array([1.0]), ())
    assert result[-1, 0] == pytest.approx(np.e, rel=1e-8)


@pytest.mark.parametrize("repeats", [0, 1, 2])
def test_zero_entropy_initialization(repeats):
    individual = SimpleNamespace(fitness=SimpleNamespace(), fitness_validation=SimpleNamespace())
    toolbox = SimpleNamespace(
        population=lambda n: [individual] * n,
        map=map,
        evaluate=lambda ind, **kw: (1, 0, 1, 0),
        compile=None,
        POP_class=lambda pop, *a, **kw: SimpleNamespace(items=pop, entropy=0),
    )
    assert len(pop_init_pheno_geno(3, toolbox, None, repeats, kwargs={})) == 3


@pytest.mark.parametrize("validation", [False, True])
def test_least_squares(validation):
    class Individual(list):
        pass

    ind = Individual(["x"])
    x = np.arange(5.0)
    fn = m.lst_matrix if validation else m.lst_matrix_noVal
    result = fn(
        ind,
        lambda *a: x,
        lambda *a, **kw: (0, 0, 0, 0),
        None,
        kwargs={"X_train": x[None, :], "y_train": 2 + 3 * x},
    )
    np.testing.assert_allclose(result[-1], [2, 3], atol=1e-12)
    assert result[0] == 0
    assert m.build_funcString([2, 3], ind) == "2+3*x"
    np.testing.assert_array_equal(
        m.evaluate_subtree(ind, 0, lambda _: lambda *a: 3, x[None, :], x), np.full(5, 3)
    )


def test_invalid_least_squares(monkeypatch):
    monkeypatch.setattr(np.linalg, "pinv", lambda a: np.full((2, 3), np.nan))
    result = m.lst_matrix(
        [1],
        lambda *a: np.ones(3),
        None,
        None,
        kwargs={"X_train": np.ones((1, 3)), "y_train": np.ones(3)},
    )
    assert len(result) == 5
    assert result[:4] == (1e6, 0, 1e6, 0)


@pytest.mark.parametrize("validation", [False, True])
def test_regression_evaluation(validation):
    x = np.arange(5.0)
    data = {"X_train": x[None, :], "y_train": 2 * x, "X_val": x[None, :], "y_val": 2 * x}
    fn = evaluate_regression if validation else evaluate_regression_noVal
    assert fn(None, lambda _: lambda x: 2 * x, kwargs=data) == (
        (0, 0, 0, 0) if validation else (0, 0)
    )
    assert fn(None, lambda _: lambda x: 0, kwargs=data)[0] == pytest.approx(np.sqrt(24))


@pytest.mark.parametrize("kind", ["IGP", "FIGP", "MGGP"])
@pytest.mark.parametrize("outputs", [1, 3])
def test_models_and_operators(kind, outputs):
    from deap import creator

    for name in ["Individual", "SubIndividual", "Fitness", "Gene"]:
        if hasattr(creator, name):
            delattr(creator, name)
    config = {
        "fitness_validation": True,
        "NgenesMax": 3,
        "stdCxpb": 0.5,
        "evaluate_regression": evaluate_regression,
    }
    pset, creator, tb = getattr(models, "define_" + kind + "_model")(
        1, outputs, 1, 1, 10, 100, 0, lambda *a, **kw: (1, 0, 1, 0), ["add", "mul"], kwargs=config
    )
    a, b = tb.population(n=2)
    if outputs > 1:
        assert len(a) == outputs
    if kind == "MGGP":
        left = a[0] if outputs > 1 else a
        right = b[0] if outputs > 1 else b
        left.w.append(1)
        assert right.w == []
    assert len(tb.mate(a, b)) == 2
    assert len(tb.mutate(a)) == 1
    assert abs(models.ephemeral_creation(2)) <= 2


@pytest.mark.parametrize("choice", [0, 0.6, 0.9])
def test_mutation_paths(choice, monkeypatch):
    pset = p.create_primitive_set(gp.PrimitiveSet("test", 1), ["add"])

    def tree():
        return gp.PrimitiveTree.from_string("add(ARG0, ARG0)", pset)

    monkeypatch.setattr(random, "random", lambda: choice)

    def expr(pset, type_):
        return gp.genFull(pset, 1, 1, type_)

    assert len(mut.xmut_MGGP([tree()], expr, pset, 0.5, 0.25)) == 1
    assert len(mut.xmut_MGGP_multiple([[tree()]], expr, pset, 0.5, 0.25)) == 1
    assert (
        len(
            mut.xmutMultiple(
                tree(), expr, 0.5, 0.25, 0.1, pset, SimpleNamespace(SubIndividual=list)
            )
        )
        == 1
    )


@pytest.mark.parametrize("probability", [0, 1])
def test_crossovers(probability):
    pset = p.create_primitive_set(gp.PrimitiveSet("test", 1), ["add"])

    def tree():
        return gp.PrimitiveTree.from_string("add(ARG0, ARG0)", pset)

    a, b = cx.xmate_MGGP([tree(), tree()], [tree(), tree()], 2, probability)
    assert 1 <= len(a) <= 2 and 1 <= len(b) <= 2
    assert len(cx.xmateMultiple([tree()], [tree()])) == 2
    assert len(cx.xmate_MGGP_multiple([[tree()]], [[tree()]], 2, probability)) == 2


def test_utilities():
    assert 1 <= len(initRepeatRandom(list, lambda: 1, 5)) <= 5
    np.testing.assert_array_equal(Min(np.array([[1, 1], [2, 0], [3, 0]])), [2, 0])

    class Fitness(base.Fitness):
        weights = (-1.0, -1.0)

    class Individual(list):
        def __init__(self, x):
            super().__init__([x])
            self.fitness = Fitness((x, 0))

    hof = HallOfFame_modified(2)
    hof.update([Individual(3), Individual(2), Individual(1)], True)
    assert hof[-1].fitness.values == (1, 0)
    assert len(list(hof)) == len(list(reversed(hof))) == len(hof.shuffle()) == 2
    assert str(hof)
    hof.remove(0)
    assert len(hof) == 1
    hof.clear()
    assert len(hof) == 0
