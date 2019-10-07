import random, copy

class Selection():
    def __init__(self, population):
        self.data     = population
        self.N        = None
        self.N_retain = None

    def _linear_normalization(self, v_min=1, v_max=10):
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

    def replace(self, pop, children):
        """
        """
        pop.sort(key=lambda x: x.aptitude, reverse=True)

        new_pop                 = copy.deepcopy(pop)
        new_pop[self.N_retain:] = children
        random.shuffle(new_pop)

        return new_pop

    def process(self, type="proportional", technique="none", gap = 0):
        """

        Args:
            type      : Selection algorithm
            technique :
            gap       :

        """

        if (technique == "elitist"):
            N_pop         = len(self.data)
            self.N        = N_pop - 1
            self.N_retain = 1

        elif (technique == "stationary"):
            N_pop   = len(self.data)
            N_child = round(N_pop*gap)

            if (N_child < 0):
                N_child = 1

            self.N        = N_child
            self.N_retain = N_pop - N_child

        elif (technique == "none"):
            N_pop         = len(self.data)
            self.N        = N_pop
            self.N_retain = 0

        else:
            raise ValueError(f"Technique {technique} not defined")

        if (type == "proportional"):
            parents = self._proportional()
        elif (type == "linear-normalization"):
            parents = self._linear_normalization()
        else:
            raise ValueError(f"Type {type} not defined")

        return parents
