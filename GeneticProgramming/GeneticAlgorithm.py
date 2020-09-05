"""
    This code explores the idea of programs that optimize through a process that mimics Natural selection.
    'The Fittest Survives' We need:
1. Problem to solve?
    - Given the necessary data, can a genetic algorithm evolve to the correct arithmetic expression that describes
      the relation between Force, Mass, and Acceleration. F=m*a
2. A fitness function aka: What makes a good solution?
    - We want to reward RETs that archive lower average divergence from the training data as well as smaller Trees.
      Fit = 1/(#Nodes) * 1/((Sum(T-h)^2)/#T)
      Refer to method fit() for more details.
3. A way to represent solutions similar to how DNA represents living organisms
    - For this script a complementary ExpressionTree.py file implements the representation of our individual
      candidates as Expression Trees (ET).
"""

from GeneticProgramming import ExpressionTree as RET
import pandas as pd
import random
import math

# Loads the training data from a csv file.
data = pd.read_csv('Sample_data.csv').values
# REF = Random Expression Forest (Collection of RET).
REF = []

#
# import matplotlib.pyplot as plt
#
# # generate axes object
# ax = plt.axes()
# # Setting up the axis.
# plt.title('Operation Fitness')
# plt.xlabel('Iterations')
# plt.ylabel('Fitness')
#
# # Sets limits
# plt.ylim(0.5, 1)
# datax, datay = [], []
#
#
# def update_plot(newdata, x_lim):
#     # Add new data to axes
#     plt.xlim(0, x_lim)
#     ax.scatter(newdata[0], newdata[1])
#     datax.append(newdata[0])
#     datay.append(newdata[1])
#     ax.plot(datax, datay)
#     # draw the plot
#     plt.draw()
#     plt.pause(0.01)  # Is necessary for the plot to update for some reason.


def _first_gen(gen_size: int, nodes_range: int, var: [any], k_range: int) -> [RET]:
    """Randomly generate the first generation of RET."""
    count = 0

    # Extend the set of arithmetic symbols to the variables allowed, and the range of constants [0,k]
    RET.ExpressionNode.extend_symbols(var, k_range)

    while count != gen_size:
        T = RET.RandomExpressionTree()
        # Randomly generate how many nodes the RET will have.
        n_nodes = random.randint(0, nodes_range)
        for i in range(n_nodes):
            op = RET.ExpressionNode.rand_symbol()
            T.insert_node(op)

        if T.valid_tree(T.root, a=10, m=2):
            REF.append(T)
            count += 1


def _fit(individual: RET.RandomExpressionTree, data):
    """
        Compute the fitness of a given tree as follows:
        Assumptions on the data: Target will be at the last column.
        m is at column 0, a is at column 1.
    """

    # Validate the RET ( Costly )
    # if not individual.valid_tree(individual.root):
    #     return 0

    sqr_difference = 0
    for sample in data:
        hypothesis = RET.RandomExpressionTree.evaluate(individual.root, a=sample[0], m=sample[1])
        sqr_difference += pow(sample[-1] - hypothesis, 2)

    avg_divergence = sqr_difference / len(data)

    fit = ( 1 / individual.nodes_count) * (1 / avg_divergence)
    fit = 1/(1 + pow(math.e, -fit))
    print(individual, fit)
    return fit

def evolve(data):

    """
        As we evolve our ET, will keep track of the fittest individuals in the population
        Will evolve by matting the fittest RET in the populations, as well as randomly mutating excising ETs.
    """

    # Mutations: May lead to invalid Trees.
    tree = REF[int(random.randint(0, len(REF)-1))]
    mutate(tree)

    result = fittest()
    RET_a, fit_a = result[0], result[1]
    result = fittest(ignore=RET_a)
    RET_b, fit_b = result[0], result[1]

    # Cross the two fittest individuals.
    cross(RET_a, RET_b)

    result = fittest()
    return result


def fittest(ignore=None):
    """
        Returns the fittest ET in the population.
        if ignore is not None then the RET referenced by ignore will not be considered.
    """
    most_fit = 0
    fittest_RET = None
    # Find the two fittest individual RETs.
    for RET_x in REF:
        if RET_x is ignore:
            continue
        # Compute fitness of this RET given the data.
        x_fit = _fit(RET_x, data)
        if x_fit > most_fit:
            fittest_RET = RET_x
            most_fit = x_fit

    return fittest_RET, most_fit


def cross(candidate_1: RET.RandomExpressionTree, candidate_2: RET.RandomExpressionTree):
    """
        Perform a cross of the candidates to generate two new individuals towards the population of ET
        To archive this, will randomly choose a node on candidate_1 and make this node become a random node on the
        candidate_2 tree, effectively generating a new tree (which is always valid).
        The same process will be done from candidate_2 to candidate_1.
    """

    candidate_node_a = _select_node(candidate_1)
    candidate_node_b = _select_node(candidate_2)

    temp_node = RET.ExpressionNode(symbol='')
    temp_node._copy(candidate_node_a)

    candidate_node_a._copy(candidate_node_b)
    candidate_node_b._copy(temp_node)



def _select_node(candidate_tree):
    steps = random.randint(0, 8)
    candidate_node = candidate_tree.root
    if steps == 0:
        return candidate_node

    else:
        for i in range(steps):
            q = random.random()
            if q >= 0.5:
                if candidate_node.right is None:
                    return candidate_node
                candidate_node = candidate_node.right
            else:
                if candidate_node.left is None:
                    return candidate_node
                candidate_node = candidate_node.left

    return candidate_node


def mutate(candidate: RET.RandomExpressionTree):

    node = _select_node(candidate)
    temp_node = RET.ExpressionNode('')
    temp_node._copy(node)

    k = random.randint(0,len(RET.ExpressionNode.operations)-1)
    node.symbol = RET.ExpressionNode.operations[int(k)]

    if not candidate.valid_tree(candidate.root):
        print('Invalid Mutation')
        node._copy(temp_node)


def print_all():
    for RET_x in REF:
        print(RET_x)


###
# Execute the Genetic Algorithm.
# 1. Generate the first generation of RETs.
_first_gen(gen_size=1000, nodes_range=8, var=['m', 'v'], k_range=0)

# For graphing purposes.
iterations = 0
x_lim = 10
# 2. Evolve the RETs.

while True:
    RET_x, fit = evolve(data)
    iterations += 1
    if iterations == x_lim:
        x_lim += 5
    # update_plot([iterations, fit], x_lim)

    # Every 100 iterations, add new RETs with some constants.
    if iterations%100 == 0:
        _first_gen(gen_size=10, nodes_range=8, var=[], k_range=5)
    if fit >= 0.99:
        # Get Result:
        print(RET_x)
        # Keep showing the graph.
        while True:
            i = 0


