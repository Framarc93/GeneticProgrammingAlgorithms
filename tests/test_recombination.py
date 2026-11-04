import pytest
from deap import base, tools

from genetic_programming_algorithms import recombination_functions as recombination


class Fitness(base.Fitness):
    weights = (-1.0,)


class Individual(list):
    def __init__(self, value):
        super().__init__([value])
        self.fitness = Fitness((value,))
        self.fitness_validation = Fitness((value,))


def setup_crossover(monkeypatch):
    parents = [Individual(1), Individual(10)]
    toolbox = base.Toolbox()

    def mate(first, second):
        # Reverse the quality ordering: the better parent's child is now worse.
        first[0], second[0] = 100, 0
        return first, second

    toolbox.register("mate", mate)
    monkeypatch.setattr(recombination.random, "sample", lambda population, k: parents)
    return parents, toolbox


def test_crossover_quality_is_decided_after_evaluation(monkeypatch):
    parents, toolbox = setup_crossover(monkeypatch)
    children = recombination.varOr(parents, toolbox, 2, 1.0, 0.0)
    assert [child[0] for child in children] == [100, 0]
    assert all(not child.fitness.valid for child in children)
    assert all(not child.fitness_validation.valid for child in children)
    for child in children:
        child.fitness.values = (child[0],)
    assert tools.selBest(children, 1)[0][0] == 0
    assert [parent[0] for parent in parents] == [1, 10]
    assert [parent.fitness.values for parent in parents] == [(1,), (10,)]
    assert all(parent.fitness_validation.valid for parent in parents)


@pytest.mark.parametrize("last_child", [0, 1])
@pytest.mark.parametrize("count", [1, 3, 5])
def test_odd_offspring_count_uses_random_child(monkeypatch, last_child, count):
    parents, toolbox = setup_crossover(monkeypatch)
    monkeypatch.setattr(recombination.random, "choice", lambda children: children[last_child])
    children = recombination.varOr(parents, toolbox, count, 1.0, 0.0)
    assert len(children) == count
    assert children[-1][0] == (100, 0)[last_child]
    assert all(not child.fitness.valid for child in children)
    assert all(child is not parent for child in children for parent in parents)
