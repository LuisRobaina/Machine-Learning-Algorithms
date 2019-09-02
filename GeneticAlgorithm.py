"""
Will explore the idea of programs that optimize through a process that mimics Natural selection. 'The Fittest Survives'
We need:
1. Problem to solve ?
2. A fitness function. What makes a good solution?
3. A way to represent solutions similar to how DNA represents living organisms.
"""

# Read the data from a cvs file
import pandas as pd
import random

data = pd.read_csv('data.csv')

# Our programs will have a set of symbols they can use
# A valid program will have a pattern such as [m,x,a,+,m]
# which translates to F = (m*a)+m (Which is incorrect).

def _first_gen(size: int) -> [any]:
    """ Randomly generate the first generation of operations"""
    pass

def _fit(individual):
    pass

def _isoperation(symbol):
    pass
def _valid_operation(operation):
    pass
def evolve(data):
    """
    As we evolve our operations will keep track of the fittest individual in the population
    Will evolve by randomly matting individuals in the populations, as well as randomnly mutating excisting individuals.
    """
    pass
    # Find the two fittest individuals, cross them and add the new individuals to the population.


