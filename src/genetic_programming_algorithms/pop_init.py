from functools import partial
from copy import copy

def pop_init_geno(size_pop, toolbox, creator, init_repeat, **kwargs):

    best_pop = toolbox.POP_class(toolbox.population(n=size_pop), creator)
    old_entropy = best_pop.entropy_len
    best_pop = best_pop.items

    for i in range(init_repeat):
        pop = toolbox.POP_class(toolbox.population(n=size_pop), creator)
        if pop.entropy_len > old_entropy and len(pop.indexes_len) == len(pop.categories_len) - 1:
            best_pop = copy(pop.items)
            old_entropy = copy(pop.entropy_len)

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


def load_population(n_iter, toolbox, creator, hof, settingsGP):
    """
    Builds the initial population starting from a population produced as output of a previous evolution

    INPUT:
        population_size,    int: size of the population
        frac_to_pass,     float: fraction of the loaded population to be passed (0, 1)
        n_iter,             int: number of iteration of the algorithm to generate the remaining individuals for the initial population
        path_pop,        string: path to the population
        init_gen            int: generation of the precomputed evolution to be used as baseline to initialze the current population
        toolbox,          class: GP standard class
        creator,          class: GP standard class

    OUPUT:
        best_population,   list: GP individuals

    Change Log:

    - 2025/05/30 - Riccardo Cadamuro, DLR intern, initial release
    """

    population_size = settingsGP["population_size"]
    frac_to_pass = settingsGP["population_fraction_to_copy"]
    path_pop = settingsGP["path"]["load_population"]
    init_gen = settingsGP["initial_generation"]

    init_population = population_size * [None]

    # compute the number of individuals to be passed
    if frac_to_pass <= 0 or frac_to_pass > 1:
        raise Exception("ERROR: frac_to_pass shall be in the range (0, 1] ")

    num_to_pass = int(population_size * frac_to_pass)

    # load the precomputed population
    path_pop = path_pop + "\Full_population_{}".format(init_gen)
    population = toolbox.POP_class(np.load(path_pop, allow_pickle=True), creator)
    best_population_load = population.items
    n_ind_load = len(best_population_load)

    if n_ind_load < num_to_pass:
        num_to_pass = n_ind_load

    pop_load_sorted = sort_individuals(best_population_load)
    init_population[:num_to_pass] = pop_load_sorted[:n_ind_load]  # copy the best <num_to_pass> individuals

    num_to_compute = population_size - num_to_pass

    if num_to_compute > 0:
        best_population_comp = init_population_max_entropy(n_iter, toolbox, creator, settingsGP)
        init_population[num_to_pass:] = best_population_comp

    best_population = hof(population_size)
    best_population.update(init_population, for_feasible=False)
    init_population = copy(best_population.shuffle())

    for i in range(population_size):
        del init_population[i].fitness.values

    return init_population