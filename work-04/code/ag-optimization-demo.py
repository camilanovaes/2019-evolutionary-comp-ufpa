import ag.chromosome, ag.selection, ag.operators
import numpy as np
import time

# Define datasets
init_pop    = list()
generations = list()
experiments = list()

# Define evaluated function
def f(x,y):
    temp_1 = np.sin(np.sqrt(x**2 + y**2))
    temp_2 = 1 + 0.001 * (x**2 + y**2)
    return 0.5 - ((temp_1**2 - 0.5)/(temp_2**2))

# Configuration parameters
config = {"f"              : f,
          "max"            : 100,
          "min"            : -100,
          "precision_bits" : 25}

# General Parameters
N_pop      = 100    # Population size
chrom_size = 50     # Chromosome size
N_gen      = 100    # Number of generations
N_exp      = 50     # Number of experiments
tc         = 0.75   # Crossover rate
tm         = 0.01   # Mutation rate
N_children = N_pop  # Number of new children

# Generate the initial population
for i in range(N_pop):
    chrom = ag.chromosome.Chromosome(chrom_size, config)
    init_pop.append(chrom)


# Run Experiments
for j in range(N_exp):
    # Make a copy of initial population
    pop = init_pop.copy()

    # Run generations
    for i in range(N_gen):
        # Apply selection
        selection = ag.selection.Selection(pop)
        parents   = selection.process(N=N_children, type="proportional")

        # Apply genetic operators: Crossover and mutation
        operator  = ag.operators.Operator(tc=tc, tm=tm)
        children  = operator.process(parents,
                                     crossover="single-point",
                                     mutation="binary-inversion")

        # Replace the entire population with the children
        pop       = children

        # Save the generation results
        generations.append(pop)

    # Save the experiments results
    experiments.append(generations)

