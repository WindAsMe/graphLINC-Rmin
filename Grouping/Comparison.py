import numpy as np
from cec2013lsgo.cec2013 import Benchmark

def CCDE(N):
    groups = []
    for i in range(N):
        groups.append([i])
    return groups


def DECC_DG(func_num, N):
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
                if i < len(groups)-1 and j < len(groups) and not DG_Differential(groups[i][0], groups[j][0], delta1, function, intercept):
                    groups[i].extend(groups.pop(j))
                    j -= 1

    return groups, cost


def DG_Differential(e1, e2, a, function, intercept):
    index1 = np.zeros(1000)
    index2 = np.zeros(1000)
    index1[e2] = 1
    index2[e1] = 1
    index2[e2] = 1

    b = function(index1) - intercept
    c = function(index2) - intercept

    return np.abs(c - (a + b)) < 0.001