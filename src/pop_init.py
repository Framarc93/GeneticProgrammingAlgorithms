from src.pop_classes import POP_geno
from functools import partial
from copy import copy

def pop_init_geno(size_pop, toolbox, creator, init_repeat, **kwargs):

    best_pop = []
    old_entropy = 0

    for i in range(init_repeat):
        pop = toolbox.POP_class(toolbox.population(n=size_pop), creator)
        if pop.entropy_len > old_entropy and len(pop.indexes_len) == len(pop.categories_len) - 1:
            best_pop = pop.items
            old_entropy = pop.entropy_len

    return best_pop


def pop_init_pheno_geno(size_pop, toolbox, creator, init_repeat, **kwargs):

    best_pop = []
    old_entropy = 0

    for i in range(init_repeat):
        pop = toolbox.population(n=size_pop)

        fitnesses = toolbox.map(partial(toolbox.evaluate, compile=toolbox.compile, kwargs=kwargs['kwargs']), pop)

        # assign evaluated fitness to population

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit[:2]
            ind.fitness_validation.values = fit[2:]

        pop_pheno = copy(toolbox.POP_class(pop, creator, kwargs=kwargs['kwargs']))
        entropy = pop_pheno.entropy

        if entropy > old_entropy:
            best_pop = copy(pop_pheno.items)
            old_entropy = copy(entropy)

    return best_pop