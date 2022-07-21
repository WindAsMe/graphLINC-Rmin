import warnings
import numpy as np
from Grouping import Proposal, Comparison
from DE import DE
from cec2013lsgo.cec2013 import Benchmark
from benchmark import benchmark
from util.help_Proposal import random_Population
from util import help_Proposal
import matplotlib.pyplot as plt
from os import path


warnings.filterwarnings("ignore")


def write_obj(data, path):
    with open(path, 'a') as f:
        f.write(str(data) + ', ')
        f.write('\n')
        f.close()


def draw_curve(x, data, title):
    plt.plot(x, data)
    plt.title(title)
    plt.show()


if __name__ == '__main__':

    N = 1000
    this_path = path.realpath(__file__)
    '''
    Decomposition method parameter initialization
    '''
    pop_size = 5
    Max_iter_search = 100
    overlap_ignore_rate = 0.5

    '''
    DE parameter initialization
    '''
    NIND = 30
    EFs = 3000000
    trail = 1
    '''
    Benchmark initialization
    '''
    for func_num in [8]:
        bench = Benchmark()
        func_info = bench.get_info(func_num)
        func = bench.get_function(func_num)
        scale_range = [func_info['lower'], func_info['upper']]
        Population = random_Population(scale_range, N, pop_size)
        DG_groups, cost_DG = Comparison.DECC_DG(func_num, N)
        # DG_groups = Comparison.CCDE(N)
        # cost_DG = 1001000
        for i in range(trail):
            print('func: ', func_num, ' trail run: ', i+1)
            cost_FDMVM = 0
            intercept = func(np.zeros(N, dtype='double'))
            g_fit = []
            '''
            Decomposition problem optimization
            '''
            # FDMVM, cost_FDMVM, g_fit = Proposal.graphLINC_Rmin(N, func, pop_size, Max_iter_search, overlap_ignore_rate,
            #                                 scale_range, cost_FDMVM, intercept, g_fit)

            # #
            # # draw_curve(np.linspace(0, cost_FDMVM, len(g_fit)), g_fit, 'Decomposition fitness')
            #
            # Max_iter_DE_FDMVM = int((EFs - cost_FDMVM) / 1000 / NIND) - 1
            #
            # var_traces_FDMVM, obj_traces_FDMVM = DE.CC(N, NIND, Max_iter_DE_FDMVM, func, scale_range, FDMVM)
            # # draw_curve(np.linspace(cost_FDMVM, EFs, len(obj_traces_FDMVM)), obj_traces_FDMVM, 'fitness')
            # FDMVM_obj_path = path.dirname(path.dirname(this_path)) + '/TEST/Data/obj/proposal/f' + str(func_num)
            # FDMVM_cost_path = path.dirname(path.dirname(this_path)) + '/TEST/Data/cost/proposal/f' + str(func_num)
            #
            # write_obj(obj_traces_FDMVM, FDMVM_obj_path)
            # write_obj(cost_FDMVM, FDMVM_cost_path)

            # DG
            Max_iter_DE_DG = int((EFs - cost_DG) / N / NIND) - 1
            var_traces_DG, obj_traces_DG = DE.CC(N, NIND, Max_iter_DE_DG, func, scale_range, DG_groups)
            DG_obj_path = path.dirname(this_path) + '/Data/obj/DG/f' + str(func_num)
            DG_cost_path = path.dirname(this_path) + '/Data/cost/DG/f' + str(func_num)
            print(DG_obj_path)
            # write_obj(obj_traces_DG, DG_obj_path)
            # write_obj(cost_DG, DG_cost_path)
            # draw_curve(np.linspace(cost_FDMVM, EFs, len(obj_traces_DG)), obj_traces_DG, 'fitness')

