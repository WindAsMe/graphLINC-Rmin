import numpy as np


def CCDE(N):
    groups = []
    for i in range(N):
        groups.append([i])
    return groups


def DECC_DG(func, N):
    intercept = func([0] * N)
    DG_matrix = np.zeros((N, N))
    for i in range(N):
        index_i = [0] * N
        index_i[i] = 0.1
        delta_i = func(index_i) - intercept

        for j in range(i+1, N):
            index_j = [0] * N
            index_ij = [0] * N
            index_j[j] = 0.1
            index_ij[i] = 0.1
            index_ij[j] = 0.1

            delta_j = func(index_j) - intercept
            delta_ij = func(index_ij) - intercept

            if np.abs(delta_ij - (delta_i + delta_j)) > 0.001:
                DG_matrix[i][j] = 1
                DG_matrix[j][i] = 1
    return DG_matrix