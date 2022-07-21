import numpy as np


def base_fitness(Population, func, intercept):
    fitness = []
    for indi in Population:
        fitness.append(func(indi) - intercept)
    return fitness


def group_individual(group, individual):
    part_individual = np.zeros(len(individual))
    for element in group:
        part_individual[element] = individual[element]
    return part_individual


def groups_fitness(groups, Population, func, cost, intercept):
    fitness = []
    for indi in Population:
        indi_fitness = 0
        for group in groups:
            indi_fitness += (func(group_individual(group, indi)) - intercept)
            cost += 1
        fitness.append(indi_fitness)
    return fitness, cost


# outer interface
# opt_fitness is calculated by groups_fitness
def object_function(base_fitness, opt_fitness):
    error = 0
    for i in range(len(base_fitness)):
        error += ((base_fitness[i] - opt_fitness[i])) ** 2
        print(error)
    return error

