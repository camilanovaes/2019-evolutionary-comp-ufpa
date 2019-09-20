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
# - [x] Debugar o método de seleção e cruzamento: Seleção (agr estamos selecionando todos os pais de uma vez ao invés de selecionar dois e fazer o cruzamento), o cruzamento tá errado, tenho que selecionar dois pais, testar a taxa de cruzamento (<75%) para cada pai e cruzar usando um número aleatório para o ponto de cruzamento.
# - [x] Usar mesma população inicial para todos os experimentos.
# - [x] Arrumar a mutação -> Tem que ser feita gene a gene.
# - [x] Arrumar o cruzamento
# - [x] Rodar 50 experimentos
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
import time, argparse
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

    return parents

def crossover(parents, tc, tm):
    """ Create children using single-point crossover

    Args:
        parents: Matrix with selected parents
        tc     : Crossover rate
        tm     : Mutation rate

    Returns:
        Vector with generated children

    """
    n_genes   = parents.shape[1]
    children  = parents.copy()

    for i in range(0, len(parents), 2):
        # Select a random point for single-point crossover
        point = np.random.randint(0, n_genes - 1)

        # Select a random number for each parent and compare with the
        # crossover rate, if both are lower than the crossover rate
        # apply the crossover. Else, just pass the parents for the new
        # population.
        tc_parent_1 = np.random.randint(0,100)
        tc_parent_2 = np.random.randint(0,100)

        if ( tc_parent_1 < tc*100 and tc_parent_2 < tc*100):
            # Children 1
            children[i, :point]   = parents[i+1, :point]
            mutation(children[i, :-1], tm)
            children[i,-1]        = fitness(children[i,:-1])

            # Children 2
            children[i+1, :point] = parents[i, :point]
            mutation(children[i+1, :-1], tm)
            children[i+1, -1]     = fitness(children[i+1, :-1])

    return children

def mutation(children, tm):
    """ Apply mutation

    Args:
        population : Matrix with all population
        tm         : Mutation rate

    """
    n_genes  = len(children)
    mutation = np.random.sample(n_genes)

    for i in range(0, n_genes):
        if (mutation[i] < tm):
            children[i] = 0 if children[i] else 1

def plot_population(pop, title, exp=0, gen=0):
    x = [decode(p) for p in pop[:, 0:25]]
    y = [decode(p) for p in pop[:, 25:50]]

    x_axis = np.arange(-100,100,0.1)
    y_axis = np.arange(-100,100,0.1)

    X,Y = np.meshgrid(x_axis,y_axis)
    Z   = f6(X, Y)

    plt.figure()
    plt.title(title)
    plt.axis((-100,100,-100,100))
    plt.contourf(X, Y, Z, 50)
    plt.scatter(x,y, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"plots/population_gen_{gen}_exp_{exp}", dpi=300)

def plot_function():
    x_axis = np.arange(-10,10,0.1)
    y_axis = np.arange(-10,10,0.1)

    X,Y = np.meshgrid(x_axis,y_axis)
    Z   = f6(X, Y)

    fig  = plt.figure()
    ax  = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    plt.savefig("plots/function", dpi=300)

def plot_fitness_vs_gen(result, type, N, title, file, x_label):

    if (type == "exp-mean"):
        mean = list()

        for i, exp in enumerate(result):
            mean.append(np.mean(exp))

        plt.figure()
        plt.title(title)
        plt.plot(np.arange(1,N), mean, marker='o', linewidth=0.5, markersize=1)
        plt.xlabel(x_label)
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(file, dpi=300)

    elif (type == "exp-best"):
        best = list()

        for i, exp in enumerate(result):
            best.append(np.amax(exp))

        plt.figure()
        plt.title(title)
        plt.plot(np.arange(1,N), best, marker='o', linewidth=0.5, markersize=1)
        plt.xlabel(x_label)
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(file, dpi=300)

    elif (type == "gen-mean"):
        plt.figure()
        plt.title(title)
        mean_mean = list()

        for i, exp in enumerate(result):
            plt.plot(np.arange(1,N), exp, marker='o', linewidth=0.5, markersize=1)

        for i in range(0, N-1):
            mean_mean.append(np.mean(result[:,i]))

        plt.plot(np.arange(1,N), mean_mean, marker='o', linewidth=1, markersize=1, color='black')

        plt.xlabel(x_label)
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(file, dpi=300)

    elif (type == "gen-best"):
        plt.figure()
        plt.title(title)
        mean_best = list()

        for i, exp in enumerate(result):
            plt.plot(np.arange(1,N), exp, marker='o', linewidth=0.5, markersize=1)

        for i in range(0, N-1):
            mean_best.append(np.mean(result[:,i]))

        plt.plot(np.arange(1,N), mean_best, marker='o', linewidth=1, markersize=1, color='black')

        plt.xlabel(x_label)
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(file, dpi=300)

def main():
    parser = argparse.ArgumentParser(description="AG")
    parser.add_argument('-f', '--file')
    parser.add_argument('--function')
    args     = parser.parse_args()

    if (not args.file):
        n_gen    = 100
        n_exp    = 50
        init_pop = initialize(100)
        plot_population(init_pop, 'População Inicial')

        gen_best = list()
        gen_mean = list()

        exp_random  = 40

        for j in range(0, n_exp):
            g_best = list()
            g_mean = list()

            # Every experiment will have the same initial population
            pop = init_pop.copy()

            for i in range(0, n_gen):
                print(f"Exp:{j} \t Generation:{i}")
                parents   = selection(pop)
                children  = crossover(parents, 0.75, 0.01)

                # The new population will be composed just with the childrens
                pop       = children

                # Generation results
                g_best.append(np.amax(pop[:, -1]))
                g_mean.append(np.mean(pop[:, -1]))

                # Plot the population in a random experiment
                if (j == exp_random and i in [0, 9, 49, 99]):
                    plot_population(pop, f"População da geração {i + 1}, experimento {j + 1}", exp=j, gen=i)

            # Generations results
            gen_best.append(g_best)
            gen_mean.append(g_mean)

        filename = "ag-" + time.strftime("%Y%m%d-%H%M%S") + ".npz"
        np.savez_compressed(filename,
                            gen_best=gen_best, gen_mean=gen_mean)

    else:
        fd       = np.load(args.file)

        gen_best = fd['gen_best']
        gen_mean = fd['gen_mean']

        plot_fitness_vs_gen(gen_best, type='gen-best', N=101, title="Melhores indivíduos por geração nos 50 experimentos",
                            file="plots/fitness_vs_exp_best", x_label="Gerações")
        plot_fitness_vs_gen(gen_mean, type='gen-mean', N=101, title="Média da população por geração nos 50 experimentos",
                            file="plots/fitness_vs_exp_mean", x_label="Gerações")

        plot_fitness_vs_gen(gen_best, type='exp-best', N=51, title="Melhor indivíduo por experimento",
                            file="plots/fitness_vs_best", x_label="Experimentos")
        plot_fitness_vs_gen(gen_mean, type='exp-mean',N=51, title="Média da população por experimento",
                            file="plots/fitness_vs_mean", x_label="Experimentos")

        plot_function()

if __name__ == "__main__":
    main()

