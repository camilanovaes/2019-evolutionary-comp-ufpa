import ag.chromosome
import ag.selection
import ag.operators
import ag.analyser
import numpy as np
import time, copy
import random

# Define evaluated function
def f6(x,y):
    temp_1 = np.sin(np.sqrt(x**2 + y**2))
    temp_2 = 1 + 0.001 * (x**2 + y**2)
    return 0.5 - ((temp_1**2 - 0.5)/(temp_2**2))

def f6_e(x,y):
    temp_1 = np.sin(np.sqrt(x**2 + y**2))
    temp_2 = 1 + 0.001 * (x**2 + y**2)
    return 999.5 - ((temp_1**2 - 0.5)/(temp_2**2))

# Configuration parameters
config = {"f"              : f6,
          "max"            : 100,
          "min"            : -100,
          "precision_bits" : 25,
          "rep_type"       : "binary"}

# General Parameters
N_pop      = 100    # Population size
chrom_size = 50     # Chromosome size
N_gen      = 100    # Number of generations
N_exp      = 1     # Number of experiments
tc         = 0.75   # Crossover rate
tm         = 0.01   # Mutation rate
gap        = 0.2    # Stationary rate
N_epoch    = 1

for k in range(N_epoch):
    # Define datasets
    init_pop    = list()
    experiments = list()

    # Generate the initial population
    for i in range(N_pop):
        chrom = ag.chromosome.Chromosome(chrom_size, config)
        init_pop.append(chrom)

    # Run Experiments
    for j in range(N_exp):
        print(f'Experiment: {j}')
        # Make a copy of initial population
        pop         = copy.deepcopy(init_pop)

        # Initialize generation list
        generations = []

        # Run generations
        for i in range(N_gen):
            # Save the population
            generations.append(pop)

            # Apply selection
            selection = ag.selection.Selection(pop)
            parents   = selection.process(type="proportional",
                                          technique="elitist", gap=gap)

            # Apply genetic operators: Crossover and mutation
            operator  = ag.operators.Operator(config=config, tc=tc, tm=tm)
            children  = operator.process(parents,
                                         crossover = "point",
                                         n_point   = 1,
                                         mutation  = "binary-inversion")

            # Replace the population with the generated children
            pop       = selection.replace(pop, children)

        # Save the experiments results
        experiments.append(generations)

    # Plot results
    analyser = ag.analyser.Analyser(experiments, epoch=k)
    analyser.plot(type="best")
    analyser.plot(type="pop")
    analyser.plot(type="mdf", show_std=False)
    analyser.plot(type="hamming", show_std=False)

    # # Plot population in diferentes generations in a random experiment
    n_exp = 40
    analyser.plot_population(config, exp=n_exp, gen=1)
    analyser.plot_population(config, exp=n_exp, gen=10)
    analyser.plot_population(config, exp=n_exp, gen=50)
    analyser.plot_population(config, exp=n_exp, gen=100)

