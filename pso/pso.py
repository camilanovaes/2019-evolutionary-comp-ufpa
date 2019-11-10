import numpy as np
from random import randint, uniform
import copy
import pdb


class Particle():
    def __init__(self, n_dim, bounds):
        self.position      = None
        self.velocity      = None
        self.best_p        = None
        self.best_f        = 0
        self.fitness       = 0
        self.n_dim         = n_dim
        self.bounds        = bounds
        self.neighbourhood = None

        # Initialize velocity and position
        lb = self.bounds[0]
        ub = self.bounds[1]

        self.position = np.random.uniform(lb, ub, size=(n_dim))
        self.velocity = np.zeros(n_dim)
        self.best_p   = self.position

    def __str__(self):
        return f'x: {self.position}; v: {self.velocity}; f: {self.fitness}'


def f6(x, y):
    temp_1 = np.sin(np.sqrt(x**2 + y**2))
    temp_2 = 1 + 0.001 * (x**2 + y**2)
    return 0.5 - ((temp_1**2 - 0.5)/(temp_2**2))

def evaluate(f, particle):
    z        = 0
    n_dim    = particle.n_dim

    for i in range(0, n_dim - 1):
        X  = particle.position[i]
        Y  = particle.position[i+1]
        z += f(X,Y)

    particle.fitness = z

    # Check to see if the current position is an individual best
    if (particle.fitness > particle.best_f):
        particle.best_p = particle.position
        particle.best_f = particle.fitness

def update_velocity(swarm, particle):
    # Parameters
    w  = 0.5
    c1 = 1
    c2 = 2
    r1 = np.random.sample(particle.n_dim)
    r2 = np.random.sample(particle.n_dim)
    v  = particle.velocity
    p  = particle.best_p
    x  = particle.position

    neighbourhood = particle.neighbourhood

    if (neighbourhood == None):
        best_p = sorted(swarm, key=lambda x: x.fitness, reverse=True)[0]
        g      = best_p.position

    else:
        neighbours = np.array(swarm)[(neighbourhood,)]
        best_p     = sorted(neighbours, key=lambda x: x.fitness, reverse=True)[0]
        g          = best_p.position

    # Velocity
    new_v = w*v + c1*r1*(p-x) + c2*r2*(g-x)

    # Save new velocity
    particle.velocity = new_v

def update_position(particle):
    # Parameters
    v  = particle.velocity
    x  = particle.position
    lb = particle.bounds[0]
    ub = particle.bounds[1]

    # Position
    new_x = x + v

    # Adjust maximum and minimum position if necessary
    ux = np.where(new_x > ub)
    if (ux):
        new_x[ux]         = ub
        v[ux]             = 0

    lx = np.where(new_x < lb)
    if (lx):
        new_x[lx]         = lb
        v[lx]             = 0

    # Save new position
    particle.position = new_x

def define_neighbourhoods(swarm, topology):
    num_particles = len(swarm)

    if (topology == 'gbest'):
        # Do nothing, the neighbourhood of each particle is the entire swarm.
        pass
    elif (topology == 'ring'):
        # Define neighbourhood in a ring topology
        neighbourhood = np.zeros((num_particles, 3), dtype=int)
        for p in range(0, num_particles):
            swarm[p].neighbourhood =[
                (p-1)%num_particles,    # particle to left
                p,                      # particle itself
                (p+1)%num_particles]    # particle to right

    elif (topology == 'von_neumann'):
        # Define neighbourhood in a von neumann topology
        # Note that the number of particles needs to be perfect square.
        # Check if the number of particles is square
        n_sqrt = int(np.sqrt(num_particles))
        if not n_sqrt**2 == num_particles:
            raise Exception ("Number of particles need to be square")

        neighbourhood = np.zeros((num_particles, 5), dtype=int)
        for p in range(0, num_particles):
            swarm[p].neighbourhood = [
                p,                                             # particle
                (p - n_sqrt) % num_particles,                  # particle above
                (p + n_sqrt) % num_particles,                  # particle below
                ((p // n_sqrt) * n_sqrt) + ((p + 1) % n_sqrt), # particle to r
                ((p // n_sqrt) * n_sqrt) + ((p - 1) % n_sqrt)]  # particle to l


def main(f, bounds, n_dim, num_particles, max_exp, max_iter, max_epochs, \
         topology='gbest', verbose=False):
    """Particle Swarm Optimization main loop

    Args:
        f             : Function
        bounds        :
        n_dim         :
        num_particles :
        max_exp       :
        max_iter      :
        max_epochs    :
        topology      : Select the topology. The supported topologies are: ring
                        (two neighbours per particle), von_neumann (four
                        neighbours per particle, swam size must be square number)
                        and gbest (the neighbourhood of each particle is the
                        entire swam).
        verbose       :

    """
    # Run epochs
    swarm_epochs = []
    for _ in range(0, max_epochs):
        initial_swarm = []
        for _ in range(0, num_particles):
            initial_swarm.append(Particle(n_dim, bounds))

        # Run experiments
        swarm_exp = []
        for exp in range(0, max_exp):
            # Init swarm
            swarm  = copy.deepcopy(initial_swarm)
            # Define the neighbourhood
            define_neighbourhoods(swarm, topology)

            # Run iterations
            for i in range(0, max_iter):
                for j in range(0, num_particles):
                    evaluate(f=f6, particle=swarm[j])

                for j in range(0, num_particles):
                    # Update velocity and position
                    update_velocity(swarm, swarm[j])
                    update_position(swarm[j])

                if (verbose):
                    best = sorted(swarm, key=lambda x: x.fitness, reverse=True)[0]
                    print(f"Inter: {i}")
                    print(f"Best: x:{best.position}; f:{best.fitness}")

            swarm_exp.append(swarm)
        swarm_epochs.append(swarm_exp)

if __name__ == "__main__":
    main(f             = f6,
         bounds        = [-100,100],
         n_dim         = 2,
         num_particles = 100,
         max_epochs    = 1,
         max_exp       = 1,
         max_iter      = 100,
         topology      = 'gbest',
         verbose       = True)

"""
TODO:
     - Implementar melhorias no calculo da velocidade:
        * Redução linear da ponderação de inércia
        * Fator de constrição
     - Implementar classe de analise e plot das métricas
     - Melhorar implementação da distância euclidiana
     - Fazer os slides
     - Fazer o trabalho escrito
"""

