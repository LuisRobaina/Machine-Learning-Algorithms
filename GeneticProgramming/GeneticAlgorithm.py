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

# Load the training data from a csv file.
data = pd.read_csv('data.csv').values

# REF = Random Expression Forest (Collection of RET).
REF = []

import matplotlib.pyplot as plt

# generate axes object
ax = plt.axes()
# Setting up the axis.
plt.title('Operation Fitness')
plt.xlabel('Iterations')
plt.ylabel('Fitness')

# Sets limits
plt.ylim(0, 1)
datax, datay = [], []


def update_plot(newdata, x_lim):
    # Add new data to axes
    plt.xlim(0, x_lim)
    ax.scatter(newdata[0], newdata[1])
    datax.append(newdata[0])
    datay.append(newdata[1])
    ax.plot(datax, datay)
    # draw the plot
    plt.draw()
    plt.pause(0.01)  # Is necessary for the plot to update for some reason.


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
    sqr_difference = 0
    for sample in data:
        hypothesis = RET.RandomExpressionTree.evaluate(individual.root, m=sample[0], a=sample[1])
        sqr_difference += pow(sample[-1] - hypothesis, 2)


    avg_divergence = (1 / (sqr_difference / len(data)) )
    print('div', avg_divergence)
    fit = ( 1/individual.nodes_count ) * avg_divergence
    # Apply Sigmoid function to result.
    return 1 / (1 + math.pow(math.e, -fit))

def evolve(data):
    global iterations
    """
        As we evolve our ET, will keep track of the fittest individuals in the population
        Will evolve by matting the fittest RET in the populations, as well as randomly mutating excising ETs.
    """
    # Assume that the first two RET are the fittest.
    RET_a, RET_b = REF[0], REF[1]

    RET_a, fit_a = fittest()
    RET_b, fit_b = fittest(ignore=RET_a)
    print(RET_a, RET_b)

    overall_fittest = fit_a if fit_a >= fit_b else fit_b
    # Cross the two individuals.
    cross(RET_a, RET_b)

    return overall_fittest


def fittest(ignore=None):
    """
        Returns the fittest ET in the population.
        if ignore is not None then the RET referenced by ignore will not be considered.
    """
    most_fit = 0
    # Find the two fittest individual RETs.
    for RET_x in REF:
        if RET_x is ignore:
            continue
        x_fit = _fit(RET_x, data)
        if x_fit > most_fit:
            RET_a = RET_x
            most_fit = x_fit
    return RET_x, most_fit


def cross(candidate_1: RET.RandomExpressionTree, candidate_2: RET.RandomExpressionTree):
    """
        Perform a cross of the candidates to generate two new individuals towards the population of ET
        To archive this, will randomly choose a node on candidate_1 and make this node become a random node on the
        candidate_2 tree, effectively generating a new tree (which is always valid).
        The same process will be done from candidate_2 to candidate_1.
    """
    candidate_node_a = _select_node(candidate_1)
    candidate_node_b = _select_node(candidate_2)

    candidate_node_a._copy(candidate_node_b)
    candidate_node_b._copy(candidate_node_a)


def _select_node(candidate_tree):
    steps = random.randint(0, 8)

    candidate_node = candidate_tree.root
    if steps == 0:
        return candidate_node
    else:
        node = candidate_tree.root
        for i in range(steps):
            q = random.random()
            if q >= 0.5:
                if node.right is None:
                    candidate_node = node
                    break
                node = node.right
            else:
                if node.left is None:
                    candidate_node = node
                    break
                node = node.left

            candidate_node = node
    return candidate_node


def mutate(candidate):
    pass


###
# Execute the Genetic Algorithm.
# 1. Generate the first generation of RETs.
_first_gen(gen_size=100, nodes_range=8, var=['m', 'a'], k_range=10)

# For graphing purposes.
iterations = 0
x_lim = 10
# 2. Evolve the RETs.
while True:
    fit = evolve(data)
    iterations += 1
    if iterations == x_lim:
        x_lim += 5
    update_plot([iterations, fit], x_lim)
    if fit >= 0.99:
        break

# Get Result:
print(fittest())
