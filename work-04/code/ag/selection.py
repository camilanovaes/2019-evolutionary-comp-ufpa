import random

class Selection():
    def __init__(self, population):
        self.data = population
        self.N    = None

    def _proportional(self):
        # List for selected parents
        parents       = list()

        # Calculate the total fitness
        total_fitness = sum(chrom.fitness for chrom in self.data)

        # Run the roulette wheel
        for i in range(0, self.N):
            pick    = random.uniform(0, total_fitness)
            current = 0

            for chrom in self.data:
                current += chrom.fitness
                if current > pick:
                    parents.append(chrom)
                    break

        return parents

    def process(self, N, type="proportional"):
        """

        Args:
            N    : Number of parents to select
            type : Selection algorithm
        """
        # Defube number of parents to be selected
        self.N = N

        if (type == "proportional"):
            parents = self._proportional()
        else:
            raise ValueError(f"Type {type} not defined")

        return parents