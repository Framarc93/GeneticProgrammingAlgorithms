from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from deap import base, gp

from examples.control.benchmarks.inverted_pendulum.dynamics import dynamics_pendulum
from examples.control.benchmarks.inverted_pendulum.evaluate import evaluate_pendulum
from examples.control.benchmarks.inverted_pendulum.Plant import Pendulum
from examples.control.propagation_functions import propagate_forward
from examples.regression.data import data_handling as data
from genetic_programming_algorithms import pop_classes as pc
from genetic_programming_algorithms import pop_init as pi
from genetic_programming_algorithms.bloat_control import staticLimit_mutShrink, staticLimit_subTrees


class Fitness(base.Fitness):
    weights = (-1.0, -1.0)


class Individual(list):
    def __init__(self, n):
        super().__init__([1] * n)
        self.fitness = Fitness((float(n), 0))
        self.fitness_validation = Fitness((float(n + 1), 0))


@pytest.mark.parametrize("cls", [pc.POP_geno, pc.POP_pheno_3D_2fit])
def test_population_methods(cls, tmp_path):
    pop = [Individual(1), Individual(2), Individual(3)]
    settings = {
        "configs": {
            "cat_number_len": 2,
            "cat_number_fit_train": 2,
            "cat_number_fit_val": 2,
            "fit_scale": 0.5,
        }
    }
    summary = cls(pop, SimpleNamespace(), kwargs=settings)
    assert len(summary) == 3
    assert summary[0] == pop[0]
    assert list(summary) == pop
    assert list(reversed(summary)) == list(reversed(pop))
    assert str(summary)
    if cls is pc.POP_geno:
        summary.output_stats("test")
        columns = 4
    else:
        summary.output_stats("test", 2, 2)
        summary.plot_entropy_evol(str(tmp_path), 0, 2, 2, 2, 0.5)
        assert (tmp_path / "gen_0.jpg").exists()
        columns = 6
    summary.save_stats(np.empty((0, columns), dtype=object), str(tmp_path) + "/", 0)
    assert (tmp_path / "Population_statistics_gen_0.npy").exists()
    values, lengths = summary.retrieve_stats(np.empty((0, 4), dtype=object), [])
    assert values.shape == (1, 4) and len(lengths) == 1


def test_load_population(monkeypatch, tmp_path):
    loaded = [Individual(4), Individual(2), Individual(3)]
    assert pi.sort_individuals(loaded)[0].fitness.values[0] == 2
    monkeypatch.setattr(pi.np, "load", lambda *a, **kw: loaded)
    tb = SimpleNamespace(
        POP_class=lambda pop, *a: SimpleNamespace(items=pop),
        pop_init=lambda size, *a, **kw: [Individual(10 + i) for i in range(size)],
    )

    class Archive:
        def __init__(self, size):
            pass

        def update(self, pop, **kw):
            self.pop = pop

        def shuffle(self):
            return self.pop

    settings = {
        "population_size": 4,
        "population_fraction_to_copy": 0.5,
        "path": {"load_population": str(tmp_path)},
        "initial_generation": 0,
    }
    pop = pi.load_population(1, tb, None, Archive, settings, kwargs={})
    assert len(pop) == 4
    assert [len(x) for x in pop] == [2, 3, 10, 11]
    assert all(not x.fitness.valid for x in pop)
    settings["population_fraction_to_copy"] = 0
    with pytest.raises(Exception):
        pi.load_population(1, tb, None, Archive, settings, kwargs={})


def test_bloat_limits():
    @staticLimit_subTrees(len, 2)
    def grow(ind):
        return ([[1] * 4, [2] * 5],)

    assert all(len(tree) <= 2 for tree in grow([[1], [2]])[0])
    pset = gp.PrimitiveSet("test", 1)
    pset.addPrimitive(np.add, 2)
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG0)", pset)

    @staticLimit_mutShrink(len, 1)
    def identity(ind):
        return (ind,)

    with pytest.raises(ValueError):
        identity([tree])


def test_control():
    plant = Pendulum()
    plant.x0 = plant.xf.copy()
    plant.Npoints = 3
    plant.t_points = np.linspace(0, 0.01, 3)

    def compile(_):
        return lambda *x: 0

    traj, u, failed = propagate_forward(plant, None, compile, dynamics_pendulum)
    assert not failed and traj.shape == (3, 4) and u.shape == (3,)
    assert evaluate_pendulum(None, compile, kwargs={"plant": plant, "dynamics": dynamics_pendulum})[
        0
    ] == pytest.approx(0, abs=1e-20)
    assert propagate_forward(plant, None, compile, lambda *a: np.full(4, np.nan))[2]


@pytest.mark.parametrize("validation", [0, 0.2])
def test_dataset(monkeypatch, tmp_path, validation):
    frame = pd.DataFrame({"x": np.arange(30), "target": np.arange(30) * 2})
    monkeypatch.setattr(data, "BASE_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(data, "fetch_data", lambda bench: calls.append(bench) or frame)
    data.retrieve_dataset_pmlb("test")
    data.retrieve_dataset_pmlb("test")
    assert calls == ["test"]
    result = data.select_testcase("test", 0.2, validation, str(tmp_path))
    assert result[0] == 1 and result[1].shape[0] == 1
    assert len(result[6]) == 6
