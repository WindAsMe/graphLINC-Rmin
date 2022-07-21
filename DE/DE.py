import geatpy as ea
import numpy as np
from util import help_DE
from DE import MyProblem, templet


def CC(Dim, NIND, MAX_iteration, func, scale_range, groups):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    initial_Population = help_DE.initial_population(Dim, NIND, func, groups, scale_range)
    real_iteration = 0

    while real_iteration < MAX_iteration:
        for i in range(len(groups)):
            solution = CC_Opt(func, scale_range, groups[i], based_population, initial_Population[i])
            initial_Population[i] = solution['lastPop']
            var_trace = solution['Vars']

            for element in groups[i]:
                var_traces[real_iteration][element] = var_trace[0][groups[i].index(element)]
                based_population[element] = var_trace[0][groups[i].index(element)]
        real_iteration += 1

    obj_traces = []
    for var in var_traces:
        obj_traces.append(func(var))
    return var_traces, obj_traces


def CC_Opt(benchmark, scale_range, group, based_population, population):
    problem = MyProblem.CC_Problem(group, benchmark, scale_range, based_population)  # 实例化问题对象

    """===========================算法参数设置=========================="""
    myAlgorithm = templet.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = 2
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    solution = ea.optimize(myAlgorithm, verbose=False, outputMsg=False, drawLog=False, saveFlag=False)
    return solution

