import random
from benchmark import benchmark
from util import help_Proposal
from Grouping import optimizer


def graphFDMVM(N, func, pop_size, Max_iter_overlap, Max_iter_search, epsilon, overlap_ignore_rate, scale_range, cost):

    """
    Algorithm initialization
    """
    initial_matrix = help_Proposal.adjacent_matrix_initial(N, 'z')
    initial_connections = help_Proposal.matrix_connection(initial_matrix)
    initial_groups = help_Proposal.connections_groups(initial_connections)
    Population = help_Proposal.random_Population(scale_range, N, pop_size)
    # base fitness is a vector
    base_fitness = benchmark.base_fitness(Population, func)
    groups_fitness, cost = benchmark.groups_fitness(initial_groups, Population, func, cost)
    delta = groups_fitness[0] - base_fitness[0]
    while delta == 0:
        delta = groups_fitness[random.randint(0, pop_size-1)] - base_fitness[random.randint(0, pop_size-1)]
    current_best_obj = benchmark.object_function(base_fitness, groups_fitness, delta, epsilon) + benchmark.penalty(
                                                    len(initial_groups), epsilon)

    '''
    apply the local search
    '''
    final_connection = help_Proposal.matrix_connection(initial_matrix)
    for i in range(Max_iter_search):
        print('iter: ', i+1, ' best obj: ', current_best_obj)
        initial_matrix, current_best_obj, cost = optimizer.local_search(current_best_obj, base_fitness, Population, func,
                                                                        initial_matrix, i, Max_iter_search, Max_iter_overlap,
                                                                        overlap_ignore_rate, delta, epsilon, cost)
        final_connection = help_Proposal.matrix_connection(initial_matrix)
        # util.draw_heatmap(initial_matrix, "YlGnBu", 'initial solution')
    help_Proposal.draw_heatmap(initial_matrix, "YlGnBu", 'Best solution')
    final_groups = help_Proposal.connections_groups(final_connection)
    print('Final groups: ', final_groups)
    return final_groups, cost
