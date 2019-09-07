"""
Will explore the idea of programs that optimize through a process that mimics Natural selection. 'The Fittest Survives'
We need:
1. Problem to solve?
    - Given the necessary data, can a genetic algorithm evolve to the correct arithmetic expression that describes
      the relation between Force, Mass, and Acceleration. F=ma
2. A fitness function aka: What makes a good solution?
    - We want to reward RET that archive lower average divergence from the training data as well as smaller Trees.
      Fit = 1/(#Nodes) * 1/((Sum(T-h)^2)/#T)
      Refer to fit for more details.

3. A way to represent solutions similar to how DNA represents living organisms
    - For this script a complementary ExpressionTree.py file implements the representation of our individual
      candidates as expression Trees.
"""

from GeneticProgramming import ExpressionTree as RET
import pandas as pd
import random

# Load the training data from a csv file.
data = pd.read_csv('data.csv').values

# REF = Random Expression Forest (Collection of RET).

REF = []

import matplotlib.pyplot as plt

# generate axes object
ax = plt.axes()

# set limits
plt.xlim(0, 1000000)
plt.ylim(0, 1000000)
datax, datay = [], []


def update_plot(newdata):
    # Add new data to axes
    ax.scatter(newdata[0], newdata[1])
    datax.append(newdata[0])
    datay.append(newdata[1])
    ax.plot(datax, datay)
    # draw the plot
    plt.draw()
    plt.pause(0.01)  # Is necessary for the plot to update for some reason.


def _first_gen(gen_size: int, nodes_range: int, var: [any], k_range: int) -> [RET]:
    """Randomly generate the first generation of Expression Trees."""
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

    return (1 / individual.nodes_count) * (1 / (sqr_difference / len(data)))


def evolve(data):
    global iterations
    """
    As we evolve our operations will keep track of the fittest individual in the population
    Will evolve by matting the fittest individuals in the populations, as well as randomly mutating excising individuals.
    """
    overall_fittest = 0
    most_fit = 0
    # Assume that the first two are the fittest.
    RET_a, RET_b = REF[0], REF[1]

    # Find the two fittest individual RETs.
    for RET_x in REF:
        x_fit = _fit(RET_x, data)
        if x_fit > most_fit:
            RET_a = RET_x
            most_fit = x_fit
    overall_fittest = most_fit
    most_fit = 0
    for RET_x in REF:
        if RET_x is RET_a:
            continue
        x_fit = _fit(RET_x, data)
        if x_fit > most_fit:
            RET_b = RET_x
            most_fit = x_fit
    if overall_fittest < most_fit:
        overall_fittest = most_fit

    # Cross the two individuals.
    cross(RET_a, RET_b)
    iterations += 1
    print('Fittest =', overall_fittest)
    update_plot([iterations, overall_fittest])


def cross(candidate_1: RET.RandomExpressionTree, candidate_2: RET.RandomExpressionTree):
    """
    Perform a cross of the candidates to generate two new individuals towards the population of RET
    To archive this, will randomly choose a node on candidate_1 and make this node become a random node on the
    candidate_2 tree, effectively generating a new tree (which is always valid).
    The same process will be done from candidate_2 to candidate_1.
    """

    candidate_node_a = _select_node(candidate_1)
    candidate_node_b = _select_node(candidate_2)

    candidate_node_a._copy(candidate_node_b)
    candidate_node_b._copy(candidate_node_a)


def _select_node(candidate_tree):
    steps = random.randint(0, 5)
    print('STEPS = ', steps)
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


# Generate the first generation.
_first_gen(gen_size=1000, nodes_range=20, var=['m', 'a'], k_range=100)
iterations = 0
while True:
    evolve(data)
