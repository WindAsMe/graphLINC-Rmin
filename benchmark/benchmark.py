import numpy as np


def base_fitness(Population, func):
    fitness = []
    for indi in Population:
        fitness.append(func(indi))
    return fitness


def group_individual(group, individual):
    part_individual = np.zeros(len(individual))
    for element in group:
        part_individual[element] = individual[element]
    return part_individual


def groups_fitness(groups, Population, func, cost):
    fitness = []
    for indi in Population:
        indi_fitness = 0
        for group in groups:
            indi_fitness += func(group_individual(group, indi))
            cost += 1
        fitness.append(indi_fitness)
    return fitness, cost


# outer interface
# hybrid with penalty
# opt_fitness is calculated by groups_fitness
def object_function(base_fitness, opt_fitness, delta, epsilon=0.1):
    error = 0
    for i in range(len(base_fitness)):
        error += ((base_fitness[i] - opt_fitness[i]) / delta) ** 2
    return epsilon * error


def penalty(groups_size, epsilon=0.1):
    return (1 - epsilon) / groups_size
