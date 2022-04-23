import random
from benchmark import benchmark
from util import help_Proposal
import copy
import numpy as np


def local_search(current_best_obj, base_fitness, Population, func, matrix, current_iter, Max_iter_search,
                 Max_iter_overlap, overlap_ignore_rate, delta, epsilon, cost, intercept):

    # set the neighbor size adaptive
    neighbor_size = define_neighbors(len(matrix), current_iter, Max_iter_search)

    # set the search points size adaptive
    search_points_size = define_search_points(len(matrix), current_iter, Max_iter_search)

    # generate the search points and its neighbors
    search_points, search_neighbors = search_point_neighbor_generate(matrix, neighbor_size, search_points_size)
    # generate the candidate solution

    matrix_solution = solution_generate(matrix, search_points, search_neighbors)

    # if current_iter % 5 == 0:
    #     help_Proposal.draw_heatmap(matrix_solution, "YlGnBu", 'Solution candidate in ' + str(current_iter + 1) + ' G')

    connections_solution = help_Proposal.matrix_connection(matrix_solution)

    groups_solution = help_Proposal.connections_groups(connections_solution)

    frequency = help_Proposal.calculate_frequency(matrix_solution)
    center_vars = help_Proposal.overlap_center(frequency)

    '''
    Solve overlap problems after new solution generate by LS
    '''
    for center_var in center_vars:
        # matrix is changed in every iteration
        temp_frequency = help_Proposal.calculate_frequency(matrix_solution)
        temp_center_vars = help_Proposal.overlap_center(temp_frequency)
        if center_var not in temp_center_vars:
            continue

        overlap_cut_matrix_solution, update_obj, cost = overlap_solve(base_fitness, current_best_obj, center_var,
                                                                      Population, func, matrix_solution, epsilon,
                                                                      Max_iter_overlap, delta, overlap_ignore_rate,
                                                                      cost, intercept)
        overlap_cut_connection_solution = help_Proposal.matrix_connection(overlap_cut_matrix_solution)
        if update_obj < current_best_obj or (len(groups_solution) > len(overlap_cut_connection_solution) and update_obj
                                             == current_best_obj):
            matrix = copy.deepcopy(overlap_cut_matrix_solution)
            current_best_obj = update_obj
    if current_iter % 5 == 0:
        help_Proposal.draw_heatmap(matrix, "YlGnBu", 'Current best solution in ' + str(current_iter + 1) + ' G')
    return matrix, current_best_obj, cost


# random cut the connection for the center variable in overlap function
def overlap_solve(base_fitness, current_best_obj, center_var, Population, func, matrix, epsilon, iteration, delta, rate,
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


def define_neighbors(N, cur_iter, Max_iter):
    return int(-N / (50 * Max_iter) * (cur_iter + 1) + N / 50) + 1
    # return int((N/5)*(1 - 1 / (1 + np.exp(-0.5 * (cur_iter - Max_iter / 3))))+1)


def define_search_points(N, cur_iter, Max_iter):
    return int(-N / (50 * Max_iter) * (cur_iter + 1) + N / 50) + 1
    # return int((N/5)*(1 - 1 / (1 + np.exp(-0.5 * (cur_iter - Max_iter / 3))))+1)


# Generate the search points and neighbors for LS
def search_point_neighbor_generate(matrix, search_size, neighbor_size):
    N = len(matrix)
    search_points = []
    while len(search_points) < search_size:
        point = random.randint(0, N - 1)
        if point not in search_points:
            search_points.append(point)
    search_neighbors = []
    for i in range(search_size):
        point_neighbors = []
        while len(point_neighbors) < neighbor_size:
            candidate_neighbor = random.randint(0, N - 1)
            if candidate_neighbor != search_points[i] and candidate_neighbor not in point_neighbors:
                point_neighbors.append(candidate_neighbor)
        search_neighbors.append(point_neighbors)
    return search_points, search_neighbors


def solution_generate(matrix, points, neighbors):
    matrix_copy = copy.copy(matrix)
    for i in range(len(points)):
        for neighbor in neighbors[i]:
            if matrix_copy[neighbor][points[i]] == 1:
                matrix_copy[neighbor][points[i]] = 0
                matrix_copy[points[i]][neighbor] = 0
            else:
                matrix_copy[neighbor][points[i]] = 1
                matrix_copy[points[i]][neighbor] = 1
    return matrix_copy





