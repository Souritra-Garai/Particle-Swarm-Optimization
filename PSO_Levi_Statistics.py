import Levi_N13
import Particle_Swarm_Optimization as PSO
import numpy as np
import time

from numba import jit

@jit(nopython=True)
def my_func(vector) :

    # print(vector)

    x = vector[:, 0]
    y = vector[:, 1]

    # the negative sign denotes we want the minimum
    # fitness is maximum when value from Levi N.13 function
    # is minimum
    return - Levi_N13.f(x, y)

# inertial factor
alpha = 0.9

# learning factors
c_g, c_p = 0.4, 0.4

# population size of particles
num_particles = 100

# number of iterations to perform
n = 300

# number of times to repeat the process for statiscal average
n_stat = 100

# setting up the PSO problem
problem = PSO.Single_Objective_PSO()

problem.set_fitness_function(my_func)
problem.set_inertia_factor(alpha)
problem.set_learning_rates(c_g, c_p)
problem.set_number_of_particles(num_particles)
problem.set_search_space_limits(Levi_N13.limits)

# setting up arrays for storing global best
g_best_pos = np.zeros((n_stat, 2))
g_best_val = np.zeros((n_stat))

t = time.time()

for k in range(n_stat) :

    # begin solving
    problem.begin()

    for i in range(n) :

        # perform one iteration of PSO
        problem.iterate()
        
    problem.stop_iterations()

    x, y = problem.get_global_best()
    g_best_pos[k] = [x, y]
    g_best_val[k] = Levi_N13.f(x, y)
    
print('Time Taken : ', time.time() - t, 's')

print('\nMean')
print('Global Best Position : ', np.average(g_best_pos, axis=0))
print('Global Best Value    : ', np.average(g_best_val))

print('\nStandard Deviation')
print('Global Best Position : ', np.std(g_best_pos, axis=0))
print('Global Best Value    : ', np.std(g_best_val))