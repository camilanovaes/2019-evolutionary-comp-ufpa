import numpy as np
import matplotlib.pyplot as plt

class Analyser():
    def __init__(self, data, epoch):
        self.data  = data
        self.epoch = epoch

    def euclidian_distance(self, gen):
        euclidian = []
        for i in range(0, len(gen)-1):
            e_gen = []
            for j in range(i+1, len(gen)):
                a = np.array(gen[i].position)
                b = np.array(gen[j].position)
                e_gen.append(np.linalg.norm(a-b))
            euclidian.append(sum(e_gen))

        return np.sum(euclidian)

    def plot(self, type, description, show_mean=True, show_std=True,
             show_particles=True, results=True, save=True):
        """

        Args:
            type             : best or pop
            show_mean        :
            show_std         :
            show_chromosomes :
            save             :

        """
        plt_config = {"best":       {"title": "Melhor indivíduo",
                                     "label": "Média do melhor indivíduo",
                                     "xlabel": "Iterações",
                                     "ylabel": "Fitness"},
                      "pop" :       {"title": "Média da população",
                                     "label": "Média da média da população",
                                     "xlabel": "Iterações",
                                     "ylabel": "Fitness"},
                      "mdf" :       {"title": "Medida de diversidade no fenótipo",
                                     "label": "MDF",
                                     "xlabel": "Iterações",
                                     "ylabel": "MDF"},
                      "euclidian" : {"title": "Medida de diversidade no genótipo",
                                     "label": "Distancia de Euclidiana",
                                     "xlabel": "Iterações",
                                     "ylabel": "Distancia de Euclidiana"}}

        label  = plt_config[type]["label"]
        title  = plt_config[type]["title"]
        xlabel = plt_config[type]["xlabel"]
        ylabel = plt_config[type]["ylabel"]

        plt.figure()

        experiment = []

        for exp in self.data:
            iteration = []
            N_iter    = len(exp)

            for ite in exp:
                fitness  = [particle.fitness for particle in ite]
                position = np.array([particle.position for particle in ite])

                if (type == "best"):
                    value = np.amax(fitness)
                elif (type == "pop"):
                    value = np.mean(fitness)
                elif (type == "mdf"):
                    best  = np.amax(fitness)
                    pop   = np.mean(fitness)
                    value = pop/best
                elif (type == "euclidian"):
                    value = self.euclidian_distance(ite)
                else:
                    raise ValueError(f"Type {type} not defined")

                iteration.append(value)

            experiment.append(iteration)

            if (show_particles):
                plt.plot(np.arange(1,N_iter+1), iteration, marker='o',
                        linewidth=0.5, markersize=1)

        if (show_mean):
            mean = []
            std  = []

            for i in range(0, N_iter):
                exp = np.array(experiment)
                mean.append(np.mean(exp[:,i]))
                std.append(np.std(exp[:,i]))

            mean = np.array(mean)
            std  = np.array(std)

            plt.plot(range(1,N_iter+1), mean, marker='o',
                     linewidth=1.5, markersize=2, color='black',
                     label=label)

            if (show_std):
                plt.fill_between(range(1,N_iter+1), mean-std, mean+std,
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