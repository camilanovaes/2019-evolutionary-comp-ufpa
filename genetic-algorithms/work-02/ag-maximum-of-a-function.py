#!/usr/bin/env python
import numpy as np
import random
from math import sin, pi
import matplotlib
import matplotlib.pyplot as plt

def initialize(size):
    """ Initialize population

    Args:
        size: Size of the population

    Returns:
        Numpy matrix with the population
    """

    population = np.zeros((size,23))

    for chrom in population:
        value      = random.randint(0, 4194303)
        binary     = list(f'{value:022b}')
        chrom[:-1] = [int(d) for d in binary]
        chrom[-1]  = fitness(chrom[:-1])

    return population

def decode(chromosome):
    """ Convert binary to real in range (-1, 2) with 6 decimal precision

    Args:
        chromosome : Chromosome vector
    """

    b = ''.join(map(str, [str(int(d)) for d in chromosome]))
    x = int(b, 2)
    v = np.array(-1.0 + (x * (3/4194303))).round(decimals=6)

    return(v)

def fitness(chromosome):
    """ Calculate fitness

    Args:
        chromosome: Chromosome to calculate fitness

    Returns:
        Calculated fitness
    """

    x      = decode(chromosome)
    fitness = x * sin(10 * pi * x) + 1

    return fitness


def selection(population, tx, k):
    """ Select chromosomes to reproduce using tournament algorithm

    Args:
        population : Matrix with all population
        tx         : Selection rate
        k          : Tournament size

    Return:
        Parents selected to reproduce
    """

    size = round(len(population) * tx)

    if (size % 2 != 0 or size == 0):
        size += 1

    parents = np.zeros((size, population.shape[1]))

    for i in range(0, size):
        better_p = min(population[:,-1])
        for j in range(0, k):
            candidate = random.choice(population)
            if (candidate[-1] >= better_p):
                better_p = candidate[-1]
                select   = candidate

        parents[i] = select

    return parents

def crossover(parents):
    """ Create children using single-point crossover

    Args:
        parents: Matrix with selected parents

    Returns:
        Vector with generated children
    """

    children = parents.copy()

    for i in range(0, len(children), 2):
        point = int(np.array(random.sample(range(1, parents.shape[1] - 1), 1)))

        # Children 1
        children[i, :point]   = parents[i+1, :point]
        children[i,-1]         = fitness(children[i,:-1])

        # Children 2
        children[i+1, :point] = parents[i, :point]
        children[i+1, -1]      = fitness(children[i+1, :-1])

    return children


# In[7]:


def mutation(population, tx):
    """ Apply mutation

    Args:
        population : Matrix with all population
        tx         : Mutation rate
    """

    n_mutation = round(len(population)*tx)

    if (n_mutation < 0):
        n_mutation = 1

    for i in range(0, n_mutation):
        chromosome = random.randint(2, population.shape[0] - 1)
        locus      = random.randint(0, population.shape[1] - 2)

        population[chromosome, locus] = random.randint(0,1)
        population[chromosome, -1]    = fitness(population[chromosome, :-1])


def f(x):
    """Optimized function"""
    return(x * sin(10 * pi * x) + 1)

def run():
    gen = 150
    pop = initialize(1000)

    better = np.zeros(gen)
    mean   = np.zeros(gen)

    for i in range(0, gen):
        pop          = pop[pop[:,-1].argsort()][::-1] # Sorte in decreasing order
        parents      = selection(pop, 0.4, 3)         # Population, tx, k
        children     = crossover(parents)
        n_keep       = len(pop) - len(children)
        pop[n_keep:] = children
        mutation(pop, 0.1)

        better[i]    = pop[0, -1]
        mean[i]      = np.mean(pop[:,-1])

    # Best chromosome
    x = decode(pop[1,:-1])
    y = pop[1,-1]
    print(f'Result: x:{x:.2f}, y:{y:.2f}')

    plt.figure()
    plt.plot(np.arange(1,gen+1), better, label = 'Best')
    plt.plot(np.arange(1,gen+1), mean, label = 'Mean')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig('result-ag.eps', format='eps', dpi=300)

    x_axis = np.linspace(-1,2,len(pop))
    y_axis = [f(i) for i in x_axis]

    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.scatter(x, y, c='r', label = f'Best: x:{x:.2f}, y:{y:.2f}')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.savefig('result2-ag.eps', format='eps', dpi=300)

if __name__ == "__main__":
    run()