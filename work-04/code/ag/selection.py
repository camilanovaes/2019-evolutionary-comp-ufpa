import random

class Selection():
    def __init__(self, population):
        self.data = population
        self.N    = None

    def _linear_normalization(self, v_max=1000, v_min=0):
        """

        Args:
            v_max :
            v_min :
        """
        # Sort population in increasing order
        self.data.sort(key=lambda x: x.aptitude)

        n_pop = len(self.data)

        for i in range(1, n_pop + 1):
            self.data[i - 1].aptitude = v_min + ((v_max - v_min)/(n_pop - 1))*(i - 1)

        parents = self._proportional()

        return parents


    def _proportional(self):
        # List for selected parents
        parents       = list()

        # Calculate the total aptitude
        total_aptitude = sum(chrom.aptitude for chrom in self.data)

        # Run the roulette wheel
        for i in range(0, self.N):
            pick    = random.uniform(0, total_aptitude)
            current = 0

            for chrom in self.data:
                current += chrom.aptitude
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
        elif (type == "linear-normalization"):
            parents = self._linear_normalization()
        else:
            raise ValueError(f"Type {type} not defined")

        return parents