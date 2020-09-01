import Levi_N13
import Particle_Swarm_Optimization as PSO
import numpy as np

import matplotlib.pyplot as plt

from numba import jit

@jit(nopython=True)
def my_func(vector) :

    # print(vector)

    x = vector[:, 0]
    y = vector[:, 1]

    return - Levi_N13.f(x, y)

problem = PSO.Single_Objective_PSO()

problem.set_fitness_function(my_func)
problem.set_inertia_factor(0.85)
problem.set_learning_rates(0.6, 0.8)
problem.set_number_of_particles(100)
problem.set_search_space_limits(np.array([[-10, 10], [-10, 10]]))

problem.begin()
n = 1000
gbest = np.zeros(n)
iters = np.arange(start=1, stop=n+1)

for i in range(n) :

    problem.iterate()
    
    x, y = problem.get_global_best()
    gbest[i] = Levi_N13.f(x,y)

problem.stop_iterations()

x, y = problem.get_global_best()
print('Global Best Vector :', problem.get_global_best())
print('Fitness Value :', Levi_N13.f(x, y))

plt.plot(iters, gbest)
plt.yscale('log')
plt.show()