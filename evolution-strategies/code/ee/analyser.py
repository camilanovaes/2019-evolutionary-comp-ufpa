import numpy as np
import matplotlib.pyplot as plt

class Analyser():
    def __init__(self, data, epoch):
        """

        Args:
            data :
        """
        self.data  = data
        self.epoch = epoch

    def hamming_distance(self, gen):
        hamming = []
        for i in range(0, len(gen)+1):
            h_gen = []
            for j in range(i+1,len(gen)):
                assert len(gen[i].bits) == len(gen[j].bits)
                h_gen.append(sum(ch1 != ch2 for ch1, ch2 in zip(gen[i].bits, gen[j].bits)))
            hamming.append(sum(h_gen))

        return hamming

    def plot(self, type, description, show_mean=True, show_std=True,
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
                               "label": "Média do melhor indivíduo",
                               "xlabel": "Gerações",
                               "ylabel": "Fitness"},
                      "pop" : {"title": "Média da população",
                               "label": "Média da média da população",
                               "xlabel": "Gerações",
                               "ylabel": "Fitness"},
                      "mdf" : {"title": "Medida de diversidade no fenótipo",
                               "label": "MDF",
                               "xlabel": "Gerações",
                               "ylabel": "MDF"},
                      "hamming" : {"title": "Medida de diversidade no genótipo",
                               "label": "hamming",
                               "xlabel": "Gerações",
                               "ylabel": "Distancia de hamming"}}

        label  = plt_config[type]["label"]
        title  = plt_config[type]["title"]
        xlabel = plt_config[type]["xlabel"]
        ylabel = plt_config[type]["ylabel"]

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
                elif (type == "mdf"):
                    best  = np.amax(fitness)
                    pop   = np.mean(fitness)
                    value = pop/best
                elif (type == "hamming"):
                    value = np.sum(self.hamming_distance(gen))
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
                print(f"{label}: result = {result}")

                with open('plots/info.txt', 'a') as fd:
                    fd.write(f"epoch: {self.epoch}\n")
                    fd.write(f"{label}: fitness = {result}\n")
                    fd.write("\n")


        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ticklabel_format(useOffset=False)
        plt.legend()

        if (save):
            plt.savefig(f"plots/{self.epoch}_fitness_vs_gen_{type}_{description}", dpi=300)
        else:
            plt.show()

    def plot_population(self, config, description, exp=1, gen=1, save=True):
        """

        Args:
            config :
            exp    :
            gen    :
            save   :

        """
        f         = config['f']
        v_max     = config['max']
        v_min     = config['min']

        n_exp = exp - 1
        n_gen = gen - 1

        pop = self.data[n_exp][n_gen]

        x = [chromo.bits[0] for chromo in pop]
        y = [chromo.bits[1] for chromo in pop]

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
            plt.savefig(f"plots/{self.epoch}_population_gen_{gen}_exp_{exp}_{description}", dpi=300)
        else:
            plt.show()
