import random, copy

class Selection():
    def replace(self, pop, children):
        """
        """
        pop.sort(key=lambda x: x.aptitude, reverse=True)

        new_pop                 = copy.deepcopy(pop)
        new_pop[self.N_retain:] = children
        random.shuffle(new_pop)

        return new_pop

    def better(self, parents, children, N=0):
        """
        Args:
            data :
            N    :

        """

        newpop = parents + children
        better = sorted(newpop, key=lambda x: x.fitness, reverse=True)
        return better[:N]

    def pairwise(self, parents, children, N=0):
        """

        Args:
            data :

        """

        newpop = []
        for p, c in zip(parents, children):
            if (p.fitness > c.fitness):
                newpop.append(p)
            else:
                newpop.append(c)

        return newpop
