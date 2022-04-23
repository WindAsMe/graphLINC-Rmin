import numpy as np

from Grouping import Proposal, Comparison
from DE import DE
from cec2013lsgo.cec2013 import Benchmark
from benchmark import benchmark
from util.help_Proposal import random_Population


if __name__ == '__main__':

    N = 1000

    '''
    Decomposition method parameter initialization
    '''
    pop_size = 5
    Max_iter_overlap = 3
    Max_iter_search = 50
    epsilon = 0.1
    overlap_ignore_rate = 0.5
    cost = 0

    '''
    DE parameter initialization
    '''
    NIND = 30
    EFs = 3000000
    '''
    Benchmark initialization
    '''
    func_num = 1
    bench = Benchmark()
    func_info = bench.get_info(func_num)
    func = bench.get_function(func_num)
    scale_range = [func_info['lower'], func_info['upper']]

    Population = random_Population(scale_range, N, pop_size)
    intercept = func(np.zeros(N))

    base_fitness = benchmark.base_fitness(Population, func, intercept)

    FDMVM, cost = Proposal.graphFDMVM(N, func, pop_size, Max_iter_overlap, Max_iter_search, overlap_ignore_rate,
                                      scale_range, cost, intercept)
    print('Decomposition cost: ', cost)
    # DE max iteration update
    Max_iter_DE = int((EFs - cost) / 1000)

    var_traces, obj_traces = DE.CC(N, NIND, Max_iter_DE, func, scale_range, FDMVM)
    print(obj_traces)
