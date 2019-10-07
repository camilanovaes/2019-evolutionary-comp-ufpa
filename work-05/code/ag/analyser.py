import numpy as np
import matplotlib.pyplot as plt

class Analyser():
    def __init__(self, data):
        """

        Args:
            data :
        """
        self.data = data

    def plot_fitness_vs_gen(self, type, show_mean=True, show_std=True,
                            show_chromosomes=True, results=True, save=True):
        """

        Args:
            type             : best or pop
            show_mean        :
            show_std         :
            show_chromosomes :
            save             :

        """
        plt_config = {"best": {"title": "Melhor indivíduo",
                               "label": "Média do melhor indivíduo"},
                      "pop" : {"title": "Média da população",
                               "label": "Média da média da população"}}

        label = plt_config[type]["label"]
        title = plt_config[type]["title"]

        plt.figure()

        experiment = []

        for exp in self.data:
            generation = []
            N_gen      = len(exp)

            for gen in exp:
                fitness = [chrom.fitness for chrom in gen]
                if (type == "best"):
                    value = np.amax(fitness)
                elif (type == "pop"):
                    value = np.mean(fitness)
                else:
                    raise ValueError(f"Type {type} not defined")

                generation.append(value)

            experiment.append(generation)

            if (show_chromosomes):
                plt.plot(np.arange(1,N_gen+1), generation, marker='o',
                        linewidth=0.5, markersize=1)

        if (show_mean):
            mean = []
            std  = []

            for i in range(0, N_gen):
                exp = np.array(experiment)
                mean.append(np.mean(exp[:,i]))
                std.append(np.std(exp[:,i]))

            mean = np.array(mean)
            std  = np.array(std)

            plt.plot(range(1,N_gen+1), mean, marker='o',
                     linewidth=1.5, markersize=2, color='black',
                     label=label)

            if (show_std):
                plt.fill_between(range(1,N_gen+1), mean-std, mean+std,
                                 linestyle='-', alpha = 0.4, color='black',
                                 label='Desvio padrão')
            if (results):
                result = np.amax(mean)
                print(f"{label}: fitness = {result}")

        plt.title(title)
        plt.xlabel("Gerações")
        plt.ylabel("Fitness")
        plt.ticklabel_format(useOffset=False)
        plt.legend()

        if (save):
            plt.savefig(f"plots/fitness_vs_gen_{type}", dpi=300)
        else:
            plt.show()

    def plot_population(self, config, exp=1, gen=1, save=True):
        """

        Args:
            config :
            exp    :
            gen    :
            save   :

        """
        f     = config['f']
        v_max = config['max']
        v_min = config['min']
        b     = config['precision_bits']

        n_exp = exp - 1
        n_gen = gen - 1

        pop = self.data[n_exp][n_gen]

        x = [chromo.decoder(chromo.bits[:b]) for chromo in pop]
        y = [chromo.decoder(chromo.bits[b:]) for chromo in pop]

        x_axis = np.arange(v_min,v_max,0.1)
        y_axis = np.arange(v_min,v_max,0.1)

        X,Y = np.meshgrid(x_axis,y_axis)
        Z   = f(X, Y)

        plt.figure()
        plt.title(f"Experimento: {exp}, Geração: {gen}")
        plt.suptitle("População")
        plt.axis((v_min,v_max,v_min,v_max))
        plt.contourf(X, Y, Z, 50)
        plt.scatter(x,y, color='red')
        plt.xlabel('x')
        plt.ylabel('y')

        if (save):
            plt.savefig(f"plots/population_gen_{gen}_exp_{exp}", dpi=300)
        else:
            plt.show()

    def print_result():
        pass