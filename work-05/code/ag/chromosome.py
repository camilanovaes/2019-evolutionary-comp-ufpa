import numpy as np
import random

class Chromosome():
    def __init__(self, size, config):
        """
        """
        self.bits     = []
        self.type     = config["rep_type"]
        self.fitness  = None
        self.aptitude = None
        self.size     = size
        self.config   = config
        self.r_max    = config["max"]
        self.r_min    = config["min"]

        if (self.type == "binary"):
            for i in range(0, size):
                self.bits.append(random.randint(0, 1))
        elif (self.type == "real"):
            for i in range(0, size):
                self.bits.append(random.uniform(self.r_min, self.r_max))
        else:
            raise ValueError(f"Type {self.type} not defined")

        # Calculate fitness
        self.eval_fitness()


    def __str__(self):
        if (self.type == "binary"):
            bits = ''.join(map(str, self.bits))
        else:
            bits = self.bits

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
        z = 0

        if (self.type == "binary"):
            X = self.decoder(self.bits[0:p])
            Y = self.decoder(self.bits[p:p*2])
            z = f(X,Y)

        elif (self.type == "real"):
            for i in range(0, self.size - 1):
                X  = self.bits[i]
                Y  = self.bits[i+1]
                z += f(X,Y)

        self.fitness  = z
        self.aptitude = self.fitness
