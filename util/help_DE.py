import geatpy as ea
from DE import MyProblem
import copy


def initial_population(Dim, NIND, func, groups, scale_range):
    initial_Population = []
    for group in groups:
        problem = MyProblem.CC_Problem(group, func, scale_range, [0] * Dim)  # 实例化问题对象

        Encoding = 'RI'  # 编码方式
        NIND = NIND      # 种群规模
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
        population = ea.Population(Encoding, Field, NIND)
        population.initChrom(NIND * len(group))
        population.Phen = copy.deepcopy(population.Chrom)

        problem.aimFunc(population)
        initial_Population.append(population)
    return initial_Population


