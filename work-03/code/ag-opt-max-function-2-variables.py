#!/usr/bin/env python
# coding: utf-8

# # Genetic Algorithm for Maximize a function with 2 variables

# Calculate the maximum of:
# $$f(x) = 0.5 - \frac{[\sin(\sqrt{(x^2 + y^2)})]^2 - 0.5}{[1 + 0.001(x^2 + y^2)]^2}$$
#
# Solution: $f(0,0) = 1$, where $x,y \in [-100, 100]$

# ### Parameters
# - Representation:
#   - Binary, fixed length
#   - 5 decimal precision
#   - Domain: $x,y \in [-100, 100]$
#   - So, we need:
#   -  $k_x \geq log_2{[100 - (-100)]*10^5} = log_2{2*10^7} = 24,25$ bits
#
# - Population size: 100
#
# - Selection: Proportional Setection
# - Genetic Operators
#   - One-point crossover: 0.75
#   - Mutation: 0.01
#
# - Termination Condition
#   - Number of generations: 100

# ## TODO:
# - [x] Mudar forma de gerar a população inicial. Sortear de 0 a 1, se for maior de 0.5 setar pra 1 e se for menor setar pra 0.
# - [ ] Debugar o método de seleção e cruzamento: Seleção me parece ok, o cruzamento tá errado, tenho que selecionar dois pais, testar a taxa de cruzamento (<75%) e cruzar usando um número aleatório para o ponto de cruzamento. 
# - [ ] Usar mesma população inicial para todos os experimentos.
# - [ ] Arrumar a seleção
# - [ ] Arrumar a mutação -> Tem que ser feita gene a gene.
# - [ ] Arrumar o cruzamento
# - [ ] Rodar 50 experimentos
# - [ ] Plotar grafico com todas as médias da população.
# - [ ] Plotar gráfico com todos os melhores indivíduos.
# - [ ] Plotar grafico com a média da média do melhor indivíduo.
# - [ ] Plotar grafico com a média da média da população.
# - [ ] Plotar desvio padão do resultado no gráfico
# - [ ] Plotar a curva de nível da função e plotar os indivíduos em cima na geração 0, 10, 50, 150 para um experimento aleatório.

# ---

import numpy as np
import random
from math import sin, pi, sqrt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

def f6(x,y):
    temp_1 = np.sin(np.sqrt(x**2 + y**2))
    temp_2 = 1 + 0.001 * (x**2 + y**2)
    return 0.5 - ((temp_1**2 - 0.5)/(temp_2**2))

def initialize(size):
    """ Initialize population

    Args:
        size: Size of the population

    Returns:
        Numpy matrix with the population
    """

    population = np.zeros((size,51))

    for chrom in population:
        x_value      = np.random.randint(0,2,25)
        y_value      = np.random.randint(0,2,25)
        chrom[0:25]  = x_value
        chrom[25:50] = y_value
        chrom[-1]    = fitness(chrom[:-1])

    return population

def decode(binary):
    """ Convert binary to real in range (-1, 2) with 6 decimal precision

    Args:
        binary : Binary vector to decode
    """

    x_bin  = ''.join(str(int(d)) for d in binary)
    x_int  = int(x_bin, 2)
    x_real = np.array(-100 + (x_int * (200/(2**25 - 1)))).round(decimals=5)

    return(x_real)

def fitness(chromosome):
    """ Calculate fitness

    Args:
        chromosome: Chromosome to calculate fitness

    Returns:
        Calculated fitness
    """

    x       = decode(chromosome[0:25])
    y       = decode(chromosome[25:50])
    fitness = f6(x,y)

    return fitness

def selection(population):
    """ Select chromosomes to reproduce using proportional selection algorithm

    Args:
        population : Matrix with all population
        tx         : Selection rate

    Return:
        Parents selected to reproduce
    """

    size          = len(population)
    # pop       = population.copy()
    parents       = np.zeros([size, population.shape[1]])
    # childrens = np.zeros([size, pop.shape[1]])

    total_fitness = sum(c[-1] for c in population)

    for i in range(0, size):
        pick    = random.uniform(0, total_fitness)
        current = 0

        for chrom in population:
            current += chrom[-1]
            if current > pick:
                parents[i] = chrom
                break

        # childrens[i:i+2] = crossover(parents, tc, tm)

    return parents

def crossover(parents, tc, tm):
    """ Create children using single-point crossover

    Args:
        parents: Matrix with selected parents

    Returns:
        Vector with generated children
    """
    n_genes       = parents.shape[1]
    children      = parents.copy()

    for i in range(0, len(parents), 2):
        point = np.random.randint(0, n_genes - 1)
        crossover_rate_1 = np.random.randint(0,100)
        crossover_rate_2 = np.random.randint(0,100)

        if ( crossover_rate_1 < tc*100 and crossover_rate_2 < tc*100):
            # Continue with the crossover
            # Children 1
            print("Pai1:", children[i])
            print("Pai2:", children[i+1])
            children[i, :point]   = parents[i+1, :point]
            mutation(children[i, :-1], tm)
            children[i,-1]        = fitness(children[i,:-1])

            # Children 2
            children[i+1, :point] = parents[i, :point]
            mutation(children[i+1, :-1], tm)
            children[i+1, -1]     = fitness(children[i+1, :-1])

            print("Filho1:", children[i])
            print("Filho2:", children[i+1])
        else:
            print("Sem filhos")

    return children

def mutation(children, tm):
    """ Apply mutation

    Args:
        population : Matrix with all population
        tx         : Mutation rate
    """

    n_genes  = len(children)
    mutation = np.random.sample(n_genes)

    for i in range(0, n_genes):
        if (mutation[i] < tm):
            children[i] = 0 if children[i] else 1

def plot_population(x,y):
    plt.figure()
    plt.scatter(x,y)
    plt.show()

def main():
    gen = 100
    pop = initialize(100)

    print(f"From pop - Best - i:{np.argmax(pop[:, -1])}, v:{np.amax(pop[:, -1])}")
    fitness = [f for f in pop[:,-1]]

    x_value = [decode(b) for b in pop[:,0:25]]
    y_value = [decode(b) for b in pop[:,25:50]]
    plot_population(x_value, y_value)

    better = np.zeros(gen)
    mean   = np.zeros(gen)

    for i in range(0, gen):
        print("------------------------------------------\n")
        print(f"Gen: {i}")
        parents      = selection(pop)             # Population, tx
        children     = crossover(parents, 0.75, 0.01)
        pop          = children

        better[i]    = np.amax(pop[:, -1])
        mean[i]      = np.mean(pop[:,-1])

        x_value = [decode(b) for b in pop[:,0:25]]
        y_value = [decode(b) for b in pop[:,25:50]]

    plt.figure()
    plt.plot(np.arange(1,gen+1), better, label = 'Better')
    plt.plot(np.arange(1,gen+1), mean, label = 'Mean')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

    plot_population(x_value, y_value)

    best = np.argmax(pop[:, -1])
    x = decode(pop[best,0:25])
    y = decode(pop[best,25:50])
    z = pop[best,-1]

    print(f'Best: x:{x:.2f}, y:{y:.2f}, f(x,y):{z:.2f}')

if __name__ == "__main__":
    main()

# x_axis = np.arange(-10,10,0.1)
# y_axis = np.arange(-10,10,0.1)

# X,Y = np.meshgrid(x_axis,y_axis)
# Z   = f6(X, Y)

# plt.figure()
# plt.contour(X, Y, Z, 20, cmap=cm.jet)
# plt.show()

# fig  = plt.figure()
# ax  = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
# plt.show()

