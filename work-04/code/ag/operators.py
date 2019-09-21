import random

class Operator():
    def __init__(self, tc, tm):
        """

        Args:
            tc     : Crossover rate
            tm     : Mutation rate
        """
        self.tc = tc
        self.tm = tm

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

    def _single_point_crossover(self, parents, mutation):
        """ Create children using single-point crossover

        Args:
            parents  : Parents to apply crossover
            mutation : Mutation type

        Returns:
            Vector with generated children

        """
        N_genes  = parents[0].size # Chromosome size
        children = parents.copy()

        for i in range(0, len(parents), 2):
            # Select a random point for single-point crossover
            point  = random.randint(0, N_genes - 1)

            # Select a random number for each parent and compare with the
            # crossover rate, if both are lower than the crossover rate
            # apply the crossover. Else, just pass the parents for the new
            # population.
            tc_parent_1 = random.randint(0,100)
            tc_parent_2 = random.randint(0,100)

            if ( tc_parent_1 < self.tc*100 and tc_parent_2 < self.tc*100):
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

        return children

    def _mutation(self, child, type):
        """

        Args:
            child : Child to apply mutation
            type  : Select the mutation type

        """

        if (type == "binary-inversion"):
            self._bin_inversion_mutation(child)
        else:
            raise ValueError(f"Type {type} not defined")

    def process(self, parents, crossover="single-point", mutation="binary-inversion"):
        """ Process operators: Crossover and mutation

        Args:
            parents   : List of parents to apply crossover and mutation
            crossover : Select the crossover algorithm
            mutation  : Select the mutation algorithm
        """
        if (crossover == "single-point"):
            children = self._single_point_crossover(parents, mutation=mutation)
        else:
            raise ValueError(f"Type {type} not defined")

        return children