import ee.chromosome
import ee.selection
import ee.operators
import ee.analyser
import numpy as np
import copy

# Define evaluated function
def f6(x, y):
    temp_1 = np.sin(np.sqrt(x**2 + y**2))
    temp_2 = 1 + 0.001 * (x**2 + y**2)
    return 0.5 - ((temp_1**2 - 0.5)/(temp_2**2))

def f6_e(x, y):
    temp_1 = np.sin(np.sqrt(x**2 + y**2))
    temp_2 = 1 + 0.001 * (x**2 + y**2)
    return 999.5 - ((temp_1**2 - 0.5)/(temp_2**2))

def printi(pop):
    for p in pop:
        print(p)

# Configuration parameters
config = {"f"              : f6,
          "max"            : 100,
          "min"            : -100}

# General Parameters
N_pop      = 1000 # Populu_ro_comma_lation size
chrom_size = 10 # Chromosome size
N_gen      = 50 # Number of generations
N_exp      = 50     # Number of experiments
N_epoch    = 1

ee_type    = "u_ro_plus_1"
N_mi       = N_pop
N_lmb      = 500
ro         = 3

for k in range(N_epoch):
    # Define datasets
    init_pop    = list()
    experiments = list()

    # Generate the initial population
    for i in range(N_pop):
        chrom = ee.chromosome.Chromosome(chrom_size, config)
        init_pop.append(chrom)

    # Run Experimens
    for j in range(N_exp):
        # Make a copy of initial population
        pop         = copy.deepcopy(init_pop)

        # Initialize generation list
        generations = []

        # Run generations
        for i in range(N_gen):
            print(f"Exp:{j} \t Gen: {i}")
            # Save the population
            generations.append(pop)

            if (ee_type == "u_plus_l"):
                operator  = ee.operators.Operator(config)
                i_parents = np.random.randint(0, len(pop), N_lmb)
                parents   = [pop[i] for i in i_parents]
                children  = operator.mutation(parents)
                selection = ee.selection.Selection()
                pop       = selection.better(pop, children, N_mi)

            elif (ee_type == "u_plus_u"):
                operator  = ee.operators.Operator(config)
                children  = operator.mutation(pop)
                selection = ee.selection.Selection()
                pop       = selection.pairwise(pop, children)

            elif (ee_type == "u_plus_u_izidio"):
                operator  = ee.operators.Operator(config)
                children  = operator.mutation(pop)
                selection = ee.selection.Selection()
                pop       = selection.better(pop, children, N_mi)

            elif (ee_type == "u_ro_plus_1"):
                operator  = ee.operators.Operator(config)
                children  = operator.crossover(pop, N_lmb, ro)
                children  = operator.mutation(children)
                selection = ee.selection.Selection()
                pop       = selection.better(pop, children, N_mi)

            elif (ee_type == "u_comma_l"):
                operator  = ee.operators.Operator(config)
                i_parents = np.random.randint(0, len(pop), N_lmb)
                parents   = [pop[i] for i in i_parents]
                children  = operator.mutation(parents)
                selection = ee.selection.Selection()
                pop       = selection.better([], children, N_mi)

            elif (ee_type == "u_ro_comma_l"):
                operator  = ee.operators.Operator(config)
                children  = operator.crossover(pop, N_lmb, ro)
                children  = operator.mutation(children)
                selection = ee.selection.Selection()
                pop       = selection.better([], children, N_mi)

            else:
                pass

        # Save the experiments results
        experiments.append(generations)

# Plot results
analyser = ee.analyser.Analyser(experiments, epoch=k)
analyser.plot(type="best", description=ee_type)
analyser.plot(type="pop", description=ee_type)
analyser.plot(type="mdf", show_std=False, description=ee_type)
analyser.plot(type="euclidian", show_std=False, description=ee_type)

# Plot population in diferentes generations in a random experiment
#n_exp = 1
#analyser.plot_population(config, exp=n_exp, gen=1)
#analyser.plot_population(config, exp=n_exp, gen=10)
#analyser.plot_population(config, exp=n_exp, gen=50)
#analyser.plot_population(config, exp=n_exp, gen=100)

