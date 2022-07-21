from Grouping import Comparison
import numpy as np
from util import help_Proposal
from cec2013lsgo.cec2013 import Benchmark
from benchmark import benchmark


def DG_Differential(e1, e2, function):

    index1 = np.zeros(1000)
    intercept = function(index1)
    index2 = np.zeros(1000)
    index1[e2] = 1
    index2[e1] = 1
    a = function(index2) - intercept
    index2[e2] = 1

    b = function(index1) - intercept
    c = function(index2) - intercept
    return np.abs(c - (a + b)) < 0.001


def CCDE(N):
    groups = []
    for i in range(N):
        groups.append([i])
    return groups


def DECC_DG(func_num, N, matrix):
    cost = 2
    bench = Benchmark()
    function = bench.get_function(func_num)
    groups = CCDE(N)
    intercept = function(np.zeros((1, N))[0])

    for i in range(len(groups)-1):
        if i < len(groups) - 1:
            cost += 2
            index1 = np.zeros((1, N))[0]
            index1[groups[i][0]] = 1
            delta1 = function(index1) - intercept

            for j in range(i+1, len(groups)):
                cost += 2
                if i < len(groups)-1 and j < len(groups) and matrix[i][j] > 0.001:
                    groups[i].extend(groups.pop(j))
                    j -= 1

    return groups, cost


func_num = 4
N = 1000
DG_groups, cost = Comparison.DECC_DG(func_num, N)
print(DG_groups)
bench = Benchmark()
func_info = bench.get_info(func_num)
func = bench.get_function(func_num)
scale_range = [func_info['lower'], func_info['upper']]
adj_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(i+1, N):
        if not DG_Differential(i, j, func):
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1

DG_groups, cost = DECC_DG(func_num, N, adj_matrix)
print(DG_groups)

