import random

import numpy as np
from Grouping import benchmark, util, Comparison
from Optimizer import optimizer
from cec2013lsgo.cec2013 import Benchmark


# separable
def Ackley(vector):
    part_1 = 0
    part_2 = 0
    D = len(vector)
    for x in vector:
        part_1 += x ** 2
        part_2 += np.cos(2 * np.pi * x)
    part_1 = -0.2 * np.sqrt(part_1 / D)
    part_2 = part_2 / D
    return -20 * np.exp(part_1) - np.exp(part_2) + 20 + np.e


# separable
def Rastrigin(vector):
    part1 = 0
    for x in vector:
        part1 += x ** 2 - 10 * np.cos(2 * np.pi * x)
    return 10 * len(vector) + part1


# separable
def Sphere(vector):
    result = 0
    for x in vector:
        result += x ** 2
    return result


# fully non-separable
def Rosenbrock(vector):
    part_1 = 0
    part_2 = 0
    for i in range(len(vector) - 1):
        part_1 += 100 * (vector[i+1] - vector[i] ** 2) ** 2
        part_2 += (vector[i] - 1)
    return part_1 + part_2


# partially non-separable
def Gap_multiply(vector):
    result = 0
    for i in range(0, len(vector)-1, 2):
        result += vector[i] * vector[i+1]
    return result


# fully non-separable
def Trid(vector):
    part_1 = 0
    part_2 = 0
    for i in range(1, len(vector)):
        part_1 += (vector[i] - 1) ** 2
        part_2 += vector[i] * vector[i-1]
    return part_1 - part_2


if __name__ == '__main__':
    '''
    Parameter config
    '''
    # func_num = 1
    # bench = Benchmark()
    # info = bench.get_info(func_num)
    # scale_range = [info['lower'], info['upper']]
    # func = bench.get_function(func_num)

    N = 50
    func = Trid
    # DG_matrix = Comparison.DECC_DG(func, N)
    # util.draw_heatmap(DG_matrix, "YlGnBu", 'LINC-R/DG', )

    pop_size = 5
    Max_iter_overlap = 3
    Max_iter_search = 50
    epsilon = 0.5
    overlap_ignore_rate = 0.5
    scale_range = [-50, 50]
    '''
    Algorithm initialization
    '''
    initial_matrix = util.adjacent_matrix_initial(N, 'z')
    initial_connections = util.matrix_connection(initial_matrix)
    initial_groups = util.connections_groups(initial_connections)
    Population = util.random_Population(scale_range, N, pop_size)
    # util.draw_heatmap(initial_matrix, "YlGnBu", 'Initial solution')
    # base fitness is a vector
    base_fitness = benchmark.base_fitness(Population, func)
    groups_fitness = benchmark.groups_fitness(initial_groups, Population, func)

    delta = groups_fitness[0] - base_fitness[0]
    while delta == 0:
        delta = groups_fitness[random.randint(0, pop_size-1)] - base_fitness[random.randint(0, pop_size-1)]
    current_best_obj = benchmark.object_function(base_fitness, groups_fitness, delta, epsilon) + benchmark.penalty(
                                                    len(initial_groups), epsilon)

    frequency = util.calculate_frequency(initial_matrix)
    center_vars = util.overlap_center(frequency)
    '''
    apply the local search
    '''
    final_connection = util.matrix_connection(initial_matrix)
    final_groups = util.connections_groups(final_connection)
    for i in range(Max_iter_search):
        print('iter: ', i+1, ' best obj: ', current_best_obj)
        initial_matrix, current_best_obj = optimizer.local_search(current_best_obj, base_fitness, Population, func,
                                                          initial_matrix, i, Max_iter_search, Max_iter_overlap,
                                                          overlap_ignore_rate, delta, epsilon)
        final_connection = util.matrix_connection(initial_matrix)
        # util.draw_heatmap(initial_matrix, "YlGnBu", 'initial solution')
    util.draw_heatmap(initial_matrix, "YlGnBu", 'Best solution')
    final_groups = util.connections_groups(final_connection)
    print('Final groups: ', final_groups)






