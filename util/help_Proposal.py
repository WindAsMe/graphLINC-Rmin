import copy
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt



def random_Population(scale_range, N, size):
    Population = np.zeros((size, N), dtype='double')
    for individual in Population:
        for i in range(len(individual)):
            individual[i] = random.uniform(scale_range[0], scale_range[1])
    return Population


# adjacent matrix initialization 'm'=mean 'r'=random 'z'=zero
def adjacent_matrix_initial(N, choice):
    matrix = np.zeros((N, N))
    if choice == 'm':
        for i in range(0, N-1, 2):
            matrix[i][i+1] = 1
            matrix[i+1][i] = 1
    elif choice == 'r':
        dense = 0.01
        for i in range(0, N):
            for j in range(0, i):
                if np.random.rand() < dense:
                    matrix[i][j] = 1
                    matrix[j][i] = 1
    elif choice == 'z':
        return matrix
    return matrix


# transform adjacent matrix to connection
def matrix_connection(matrix):
    connections = []
    N = len(matrix)
    for i in range(N):
        for j in range(i+1, N):
            if matrix[i][j] == 1:
                connections.append([i, j])
    connections = element_contain(connections, N)
    return connections


def element_contain(connections, N):
    straight_connection = []
    for conn in connections:
        straight_connection.extend(conn)
    for i in range(N):
        if i not in straight_connection:
            connections.append([i])
    return connections


# transform connections to group
# sample like [[1, 2], [2, 3], [4]] to [[1, 2, 3], [4]]
def connections_groups(connections):
    copy_connections = copy.deepcopy(connections)
    merged_groups = []
    for conn in connections:
        if len(conn) == 1:
            merged_groups.append(conn)
            copy_connections.remove(conn)
    while len(copy_connections) != 0:
        new_connection = copy_connections.pop()
        element_list = copy.deepcopy(new_connection)
        while len(element_list) != 0:
            element = element_list.pop()
            for group in copy_connections:
                if element in group:
                    copy_connections.remove(group)
                    new_connection.extend(group)
                    element_list.extend(group)
        merged_groups.append(list(set(new_connection)))
    return merged_groups


# transform connections to adjacent matrix
def connections_matrix(connections, N):
    matrix = np.zeros((N, N))
    for conn in connections:
        if len(conn) > 1:
            matrix[conn[0]][conn[1]] = 1
            matrix[conn[1]][conn[0]] = 1
    return matrix


def calculate_frequency(matrix):
    frequency = [0] * len(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                frequency[i] += 1
    return frequency


def overlap_center(frequency):
    # extract the center of overlap parts
    center_vars = []
    for i in range(len(frequency)):
        if frequency[i] > 2:
            center_vars.append(i)
    return center_vars


def overlap_cut(var_center, overlap_connection, ignore_rate):
    selection = np.random.rand(len(overlap_connection))
    new_connection = []
    flag = False
    for i in range(len(overlap_connection)):
        if selection[i] <= ignore_rate:
            c = copy.deepcopy(overlap_connection[i])
            c.remove(var_center)
            new_connection.append(c)
        else:
            new_connection.append(overlap_connection[i])
            flag = True
    if not flag:
        new_connection.append([var_center])
    return new_connection


def draw_heatmap(matrix, color, title=None):
    ax = plt.axes()
    sns.heatmap(matrix, ax=ax, cmap=color,)
    if title is not None:
        ax.set_title(title)
    plt.show()
