from benchmark import benchmark
from util import help_Proposal
from Grouping import optimizer


def graphFDMVM(N, func, pop_size, Max_iter_overlap, Max_iter_search, overlap_ignore_rate, scale_range, cost,
               intercept):

    """
    Algorithm initialization
    """
    stop_threshold = 0.1
    initial_matrix = help_Proposal.adjacent_matrix_initial(N, 'z')
    initial_connections = help_Proposal.matrix_connection(initial_matrix)
    initial_groups = help_Proposal.connections_groups(initial_connections)
    Population = help_Proposal.random_Population(scale_range, N, pop_size)
    # base fitness is a vector
    base_fitness = benchmark.base_fitness(Population, func, intercept)

    groups_fitness, cost = benchmark.groups_fitness(initial_groups, Population, func, cost, intercept)
    current_best_obj = benchmark.object_function(base_fitness, groups_fitness)
    print('current best obj: ', current_best_obj)

    '''
    apply the local search
    '''
    final_connection = help_Proposal.matrix_connection(initial_matrix)
    for i in range(Max_iter_search):

        print('iter: ', i+1, ' best obj: ', current_best_obj)
        if current_best_obj < stop_threshold:
            break
        initial_matrix, current_best_obj, cost = optimizer.local_search(current_best_obj, base_fitness, Population, func,
                                                                        initial_matrix, i, Max_iter_search,
                                                                        Max_iter_overlap, overlap_ignore_rate, cost, intercept)
        final_connection = help_Proposal.matrix_connection(initial_matrix)
        # util.draw_heatmap(initial_matrix, "YlGnBu", 'initial solution')
    help_Proposal.draw_heatmap(initial_matrix, "YlGnBu", 'Best solution')
    final_groups = help_Proposal.connections_groups(final_connection)
    print('Final groups: ', final_groups)
    return final_groups, cost
