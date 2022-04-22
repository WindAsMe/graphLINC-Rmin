from Grouping import Proposal, Comparison
from DE import DE


def ADD(vector):
    return sum(vector)


def none_separable(vector):
    result = 0
    for i in range(0, len(vector)-1, 2):
        result += vector[i]*vector[i+1]
    return result


if __name__ == '__main__':

    N = 5

    '''
    Decomposition method parameter initialization
    '''
    pop_size = 5
    Max_iter_overlap = 3
    Max_iter_search = 50
    epsilon = 0.5
    overlap_ignore_rate = 0.5
    cost = 0

    '''
    DE parameter initialization
    '''
    NIND = 5
    Max_iter_DE = 10
    '''
    Benchmark initialization
    '''
    func = ADD
    scale_range = [-10, 10]
    # func_num = 1
    # bench = Benchmark()
    # func_info = bench.get_info(func_num)
    # func = bench.get_function(func_num)
    # scale_range = [func_info['lower'], func_info['upper']]

    # FDMVM, cost = Proposal.graphFDMVM(N, func, pop_size, Max_iter_overlap, Max_iter_search, epsilon, overlap_ignore_rate
    #                                   , scale_range, cost)

    FDMVM = Comparison.CCDE(N)
    var_traces, obj_traces = DE.CC(N, NIND, Max_iter_DE, func, scale_range, FDMVM)
    print(obj_traces)


