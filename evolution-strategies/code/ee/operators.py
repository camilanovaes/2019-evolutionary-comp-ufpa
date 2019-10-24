import numpy as np
import random
import copy

class Operator():
    def __init__(self, config):
        """

        Args:

        """
        self.config  = config

    def crossover(self, parents, N_lmb, ro):
        """

        Args:
            parents :
            N_lmb   :
            ro      :

        """
        children =[]
        for i in range(N_lmb):
            parent           = np.random.choice(parents, ro)
            children.append(parent[0])
            children[i].bits = np.mean([f.bits for f in parent], axis=0)
            children[i].std  = np.mean([f.std for f in parents])

        return children

    def mutation(self, parents):
        """

        Args:
            parents:

        Return:
            List of childrens

        """
        N_genes  = parents[0].size # Chromosome size
        children = [copy.deepcopy(c) for c in parents]

        for i in range (0, len(parents)):
            for j in range(N_genes):
                children[i].bits[j] += np.random.normal(0, parents[i].std)
                children[i].eval_fitness()

        return children
