import numpy as np
import random

class Chromosome():
    def __init__(self, size, config, type="binary", r_max=1, r_min=0):
        """
        """
        self.bits    = []
        self.fitness = None
        self.size    = size
        self.config  = config
        self.r_max   = r_max
        self.r_min   = r_min

        if (type == "binary"):
            for i in range(0, size):
                self.bits.append(random.randint(0, 1))
        else:
            raise ValueError(f"Type {type} not defined")

        # Calculate fitness
        self.eval_fitness()


    def __str__(self):
        bits = ''.join(map(str, self.bits))

        return f'{bits}, {self.fitness}'

    def decoder(self, binary):
        """ Convert binary to real

        Args:
            binary    : Binary value to decode
            max       : Maximum value
            min       : Minimum value
            bits      : Number of bits for precision

        Returns:
            Real value

        """
        # Configuration parameters
        max_v     = self.config["max"]
        min_v     = self.config["min"]
        precision = self.config["precision_bits"]

        x_bin  = ''.join(map(str, binary))
        x_int  = int(x_bin, 2)
        x_real = min_v + (x_int * ((max_v - min_v))/(2**precision - 1))

        return x_real

    def eval_fitness(self):
        """ Calculate fitness

        Args:
            f         : Evaluation function
            chromosome: Chromosome to calculate fitness

        Returns:
            Calculated fitness
        """
        p = self.config["precision_bits"]
        f = self.config["f"]

        X = self.decoder(self.bits[0:p])
        Y = self.decoder(self.bits[p:p*2])
        z = f(X,Y)

        self.fitness = z
