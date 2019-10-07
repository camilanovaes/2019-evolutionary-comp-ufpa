import random
import copy

class Operator():
    def __init__(self, config, tc, tm):
        """

        Args:
            tc     : Crossover rate
            tm     : Mutation rate
        """
        self.config = config
        self.tc     = tc
        self.tm     = tm

    def _bin_inversion_mutation(self, child):
        """ Just invert the bit

        Args:
            child: Child to mutate

        """
        N_genes  = child.size # Chromosome size

        for i in range(0, N_genes):
            mutation = random.randint(0, 100)

            if (mutation < self.tm*100):
                child.bits[i] = 0 if child.bits[i] else 1

    def _rand_mutation(self, child):
        N_genes = child.size            # Chromosome size
        r_max   = self.config["max"]
        r_min   = self.config["min"]

        for i in range(0, N_genes):
            mutation = random.randint(0, 100)

            if (mutation < self.tm*100):
                child.bits[i] = random.uniform(r_min, r_max)

    def _point_crossover(self, n_point, parents, mutation):
        """ Create children using n-point crossover

        Args:
            ppassarents  : Parents to apply crossover
            mutation : Mutation type

        Returns:
            Vector with generated children

        """
        N_genes  = parents[0].size # Chromosome size
        children = [copy.deepcopy(c) for c in parents]

        for i in range(0, len(parents), 2):
            # Select a random points for n-point crossover
            for j in range(n_point):
                point  = random.randint(0, N_genes - 1)

                # Select a random number for each parent and compare with the
                # crossover rate, if both are lower than the crossover rate
                # apply the crossover. Else, just pass the parents for the new
                # population.
                tc_parent_1 = random.randint(0,100)
                tc_parent_2 = random.randint(0,100)

                if ( tc_parent_1 < self.tc*100 and tc_parent_2 < self.tc*100):
                    try:
                        # Child 1
                        # Apply crossover, then mutates and finally recalculates
                        # the fitness
                        children[i].bits[:point] = parents[i+1].bits[:point]
                        self._mutation(children[i], mutation)
                        children[i].eval_fitness()

                        # Child 2
                        # Apply crossover, then mutates and finally recalculates
                        # the fitness
                        children[i+1].bits[:point] = parents[i].bits[:point]
                        self._mutation(children[i+1], mutation)
                        children[i+1].eval_fitness()

                    except IndexError:
                        pass

        return children

    def _uniform_crossover(self, parents, mutation):
        """ Create children using uniform crossover

        Args:
            parents  : Parents to apply crossover
            mutation : Mutation type

        Returns:
            Vector with generated children

        """
        N_genes  = parents[0].size # Chromosome size
        children = [copy.deepcopy(c) for c in parents]

        for i in range(0, len(parents), 2):
            # Select a random point for n-point crossover
            mask  = []
            for j in range(0, N_genes):
                mask.append(random.randint(0, 1))

            # Select a random number for each parent and compare with the
            # crossover rate, if both are lower than the crossover rate
            # apply the crossover. Else, just pass the parents for the new
            # population.
            tc_parent_1 = random.randint(0,100)
            tc_parent_2 = random.randint(0,100)
            if ( tc_parent_1 < self.tc*100 and tc_parent_2 < self.tc*100):
                try:
                    for m, k in enumerate(mask):
                        if (m):
                            # Child 1
                            children[i].bits[k] = parents[i+1].bits[k]
                        else:
                            # Child 2
                            children[i+1].bits[k] = parents[i].bits[k]

                    # Apply mutation and recalculates the fitness
                    self._mutation(children[i], mutation)
                    children[i].eval_fitness()
                    self._mutation(children[i+1], mutation)
                    children[i+1].eval_fitness()

                except IndexError:
                    pass

        return children

    def _arithmetic_crossover(self, parents, mutation):
        """ Create children using uniform crossover

        Args:
            parents  : Parents to apply crossover
            mutation : Mutation type

        Returns:
            Vector with generated children

        """
        N_genes  = parents[0].size # Chromosome size
        children = [copy.deepcopy(c) for c in parents]

        for i in range(0, len(parents), 2):
            a = random.uniform(0,1)

            # Select a random number for each parent and compare with the
            # crossover rate, if both are lower than the crossover rate
            # apply the crossover. Else, just pass the parents for the new
            # population.
            tc_parent_1 = random.randint(0,100)
            tc_parent_2 = random.randint(0,100)
            if ( tc_parent_1 < self.tc*100 and tc_parent_2 < self.tc*100):
                try:
                    for j in range(0, N_genes):
                        children[i].bits[j]   = a*children[i].bits[j] \
                                              + (1 - a)*children[i+1].bits[j]

                        children[i+1].bits[j] = a*children[i+1].bits[j] \
                                              + (1 - a)*children[i].bits[j]

                    # Apply mutation and recalculates the fitness
                    self._mutation(children[i], mutation)
                    children[i].eval_fitness()
                    self._mutation(children[i+1], mutation)
                    children[i+1].eval_fitness()

                except IndexError:
                    pass

        return children

    def _mutation(self, child, type):
        """

        Args:
            child : Child to apply mutation
            type  : Select the mutation type

        """

        if (type == "binary-inversion"):
            self._bin_inversion_mutation(child)
        elif (type == "random-uniform"):
            self._rand_mutation(child)
        else:
            raise ValueError(f"Type {type} not defined")

    def process(self, parents, crossover="point", n_point=1, mutation="binary-inversion"):
        """ Process operators: Crossover and mutation

        Args:
            parents   : List of parents to apply crossover and mutation
            crossover : Select the crossover algorithm
            mutation  : Select the mutation algorithm
        """
        if (crossover == "point"):
            children = self._point_crossover(n_point, parents, mutation=mutation)
        elif (crossover == "uniform"):
            children = self._uniform_crossover(parents, mutation=mutation)
        elif (crossover == "arithmetic"):
            children = self._arithmetic_crossover(parents, mutation=mutation)
        else:
            raise ValueError(f"Type {type} not defined")

        return children