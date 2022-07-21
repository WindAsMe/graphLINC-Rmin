import random
from benchmark import benchmark
from util import help_Proposal
import copy
import numpy as np


def local_search(current_best_obj, base_fitness, Population, func, matrix, current_iter, Max_iter_search,
                 overlap_ignore_rate, cost, intercept):

    # set the neighbor size adaptive
    neighbor_size = define_neighbors(current_iter, Max_iter_search)
    # set the search points size adaptive
    search_points_size = define_search_points(current_iter, Max_iter_search)
    # generate the search points and its neighbors
    search_paris = search_pairs_generate(matrix, neighbor_size, search_points_size)
    # generate the candidate solution
    matrix_solution = solution_generate(matrix, search_paris)

    # if current_iter % 5 == 0:
    #     help_Proposal.draw_heatmap(matrix_solution, "YlGnBu", 'Solution candidate in ' + str(current_iter + 1) + ' G')

    connections_solution = help_Proposal.matrix_connection(matrix_solution)
    groups_solution = help_Proposal.connections_groups(connections_solution)

    frequency = help_Proposal.calculate_frequency(matrix_solution)
    center_vars = help_Proposal.overlap_center(frequency)

    '''
    Solve overlap problems after new solution generate by LS
    '''
    overlap_cut_matrix_solution, update_obj, cost = easy_overlap_solve(center_vars, matrix_solution, base_fitness,
                                                                       Population, func, cost, intercept,
                                                                       overlap_ignore_rate)
    overlap_cut_connection_solution = help_Proposal.matrix_connection(overlap_cut_matrix_solution)
    if update_obj < current_best_obj or (len(groups_solution) > len(overlap_cut_connection_solution) and update_obj
                                         == current_best_obj):
        matrix = copy.deepcopy(overlap_cut_matrix_solution)
        current_best_obj = update_obj

    # for center_var in center_vars:
    #     # matrix is changed in every iteration
    #     temp_frequency = help_Proposal.calculate_frequency(matrix_solution)
    #     temp_center_vars = help_Proposal.overlap_center(temp_frequency)
    #     if center_var not in temp_center_vars:
    #         continue
    #     overlap_cut_matrix_solution, update_obj, cost = overlap_solve(base_fitness, current_best_obj, center_var,
    #                                                                   Population, func, matrix_solution,
    #                                                                   Max_iter_overlap, overlap_ignore_rate, cost,
    #                                                                   intercept)
    #
    #     overlap_cut_connection_solution = help_Proposal.matrix_connection(overlap_cut_matrix_solution)
    #     if update_obj < current_best_obj or (len(groups_solution) > len(overlap_cut_connection_solution) and
    #                                                           update_obj  == current_best_obj):
    #         matrix = copy.deepcopy(overlap_cut_matrix_solution)
    #         current_best_obj = update_obj

    return matrix, current_best_obj, cost


# Just random cut the overlap
def easy_overlap_solve(center_vars, matrix_solution, base_fitness, Population, func, cost, intercept, ignore_rate):
    overlap_cut_matrix_solution = copy.deepcopy(matrix_solution)
    overlap_connections = []
    for center_var in center_vars:
        for i in range(len(matrix_solution[center_var])):
            if matrix_solution[center_var][i] == 1:
                overlap_connections.append([center_var, i])
    random_selection = np.random.rand(len(overlap_connections))
    for i in range(len(random_selection)):
        if random_selection[i] < ignore_rate:
            overlap_cut_matrix_solution[overlap_connections[i][0]][overlap_connections[i][1]] = 0
            overlap_cut_matrix_solution[overlap_connections[i][1]][overlap_connections[i][0]] = 0

    overlap_cut_connection = help_Proposal.matrix_connection(overlap_cut_matrix_solution)
    overlap_cut_groups = help_Proposal.connections_groups(overlap_cut_connection)
    update_fitness, cost = benchmark.groups_fitness(overlap_cut_groups, Population, func, cost, intercept)
    update_obj = benchmark.object_function(base_fitness, update_fitness)
    return overlap_cut_matrix_solution, update_obj, cost


# random cut the connection for the center variable in overlap function
def overlap_solve(base_fitness, current_best_obj, center_var, Population, func, matrix, iteration, rate,
                  cost, intercept):
    # overlap_connection save the double element pair
    # overlap_group save the merged element group
    overlap_connection = []
    overlap_group = [center_var]
    for j in range(len(matrix[0])):
        if matrix[center_var][j] == 1:
            overlap_connection.append([center_var, j])
            overlap_group.append(j)

    matrix_copy = copy.deepcopy(matrix)
    groups = help_Proposal.connections_groups(help_Proposal.matrix_connection(matrix))
    pure_groups = []
    for group in groups:
        if center_var not in group:
            pure_groups.append(group)
    rest_fitness, cost = benchmark.groups_fitness(pure_groups, Population, func, cost, intercept)
    # random ignore the connection with certain rate
    for i in range(iteration):

        # merge the sample like [[1, 2], [2, 3], [4]] to [[1, 2, 3], [4]]
        # new_connection saves like [[1, 2], [2, 3], [4]]
        # new_groups saves like [[1, 2, 3], [4]]
        new_connection = help_Proposal.overlap_cut(center_var, overlap_connection, rate)
        new_groups = help_Proposal.connections_groups(new_connection)

        new_part_fitness, cost = benchmark.groups_fitness(new_groups, Population, func, cost, intercept)
        new_part_obj = benchmark.object_function(base_fitness, np.sum([new_part_fitness, rest_fitness], axis=0))
        # update
        if new_part_obj <= current_best_obj:
            current_best_obj = new_part_obj
            matrix_copy = copy.deepcopy(matrix)
            for conn in new_connection:
                if len(conn) < 2:
                    matrix_copy[center_var][conn[0]] = 0
                    matrix_copy[conn[0]][center_var] = 0
    return matrix_copy, current_best_obj, cost


# (0, 10) and (max, 2)
def define_neighbors(cur_iter, Max_iter):
    return int(-8/Max_iter * cur_iter + 10)


def define_search_points(cur_iter, Max_iter):
    return int(-8/Max_iter * cur_iter + 10)


# Generate the search points and neighbors for LS
def search_pairs_generate(matrix, search_size, neighbor_size):
    N = len(matrix)
    search_pairs = []
    for i in range(search_size * neighbor_size):
        e1 = np.random.randint(0, N-1)
        e2 = np.random.randint(0, N-1)
        while [e1, e2] in search_pairs or [e2, e1] in search_pairs:
            e1 = np.random.randint(0, N - 1)
            e2 = np.random.randint(0, N - 1)
        search_pairs.append([e1, e2])
    return search_pairs


def solution_generate(matrix, search_pairs):
    matrix_copy = copy.copy(matrix)
    for pair in search_pairs:
        if matrix_copy[pair[0]][pair[1]] == 0:
            matrix_copy[pair[0]][pair[1]] = 1
            matrix_copy[pair[1]][pair[0]] = 1
        else:
            matrix_copy[pair[0]][pair[1]] = 0
            matrix_copy[pair[0]][pair[1]] = 0
    return matrix_copy





