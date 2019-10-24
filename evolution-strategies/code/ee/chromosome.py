import numpy as np
import random

class Chromosome():
    def __init__(self, size, config):
        """
        """
        self.bits     = []
        self.std      = None
        self.fitness  = None
        self.aptitude = None
        self.size     = size
        self.config   = config
        self.r_max    = config["max"]
        self.r_min    = config["min"]

        for i in range(0, size):
            self.bits.append(random.uniform(self.r_min, self.r_max))

        # Calculate fitness
        self.eval_fitness()

        # Gen std value
        self.std = np.random.uniform(0,1)

    def __str__(self):
        bits = self.bits

        return f'{bits}, {self.fitness}'

    def eval_fitness(self):
        """ Calculate fitness

        Args:
            f         : Evaluation function
            chromosome: Chromosome to calculate fitness

        Returns:
            Calculated fitness

        """
        f = self.config["f"]
        z = 0

        for i in range(0, self.size - 1):
            X  = self.bits[i]
            Y  = self.bits[i+1]
            z += f(X,Y)

        self.fitness  = z
        self.aptitude = self.fitness
