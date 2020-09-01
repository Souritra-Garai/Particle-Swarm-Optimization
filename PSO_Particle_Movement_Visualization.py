import Levi_N13
import Particle_Swarm_Optimization as PSO
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
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
num_particles = 5

# number of iterations to perform
n = 300

# setting up the PSO problem
problem = PSO.Single_Objective_PSO()

problem.set_fitness_function(my_func)
problem.set_inertia_factor(alpha)
problem.set_learning_rates(c_g, c_p)
problem.set_number_of_particles(num_particles)
problem.set_search_space_limits(Levi_N13.limits)

# setting up the matrix to store positions
position = np.zeros((n+1, num_particles, 2))

# setting up arrays for storing global best
g_best_pos = np.zeros((n+1, 2))
g_best_val = np.zeros((n+1))
iters = np.arange(start=0, stop=n+1)

# begin solving
problem.begin()

# store the starting position
position[0, :, :] = problem.get_particle_positions()

# store starting global best
g_best_pos[0] = problem.get_global_best()
g_best_val[0] = Levi_N13.f(g_best_pos[0, 0], g_best_pos[0, 1])

for i in range(1, n+1) :

    # perform one iteration of PSO
    problem.iterate()
    
    # store the updated positions
    position[i, :, :] = problem.get_particle_positions()

    # store the updated global best values
    g_best_pos[i] = problem.get_global_best()
    g_best_val[i] = Levi_N13.f(g_best_pos[i, 0], g_best_pos[i, 1])

problem.stop_iterations()

x, y = problem.get_global_best()
print('Global Best Vector :', problem.get_global_best())
print('Fitness Value :', Levi_N13.f(x, y))

# initialize figure to be used for plotting
fig, (scat_ax, g_ax) = plt.subplots(1, 2)

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

# initialising the points
scat = scat_ax.scatter(position[0, :, 0], position[0, :, 1], c=np.random.rand(num_particles,3), s=500)
g_pos = scat_ax.scatter(g_best_pos[0, 0], g_best_pos[0, 0], c='black', edgecolors='black', s=600, label='Global Best Position', lw=3, marker='X')

# set axes limits
scat_ax.axis(Levi_N13.limits.flatten())

# set axes ticks
scat_ax.set_xticks(np.arange(-10, 11))
scat_ax.set_yticks(np.arange(-10, 11))

# set axes labels
scat_ax.set_xlabel('x', fontsize=15)
scat_ax.set_ylabel('y', fontsize=15)

# set grid
scat_ax.grid()

# show legends
scat_ax.legend(fontsize=20, loc=1)

# set title
scat_ax.set_title('Position of Particles in the Search Space', fontsize=15)

# initialising the global best vs iteration plot
g_line, = g_ax.plot(iters[:1], g_best_val[:1], lw = 2)

# setting the axes limits
g_ax.set_xlim([0, n])
g_ax.set_ylim([-0.2, 2.5])

# set axes labels
g_ax.set_xlabel('Iteration #', fontsize=15)
g_ax.set_ylabel('Global Best Value', fontsize=15)

# set title
g_ax.set_title('Evolution of Global Best with Iterations', fontsize=15)

# set title
fig.suptitle('Particle Swarm Optimization for Levi N.13 Function', fontsize=20, fontweight='bold')

def animate(j) :

    scat.set_offsets(position[j])
    g_pos.set_offsets(g_best_pos[j])

    g_line.set_data(iters[:j+1], g_best_val[:j+1])

    return scat, g_pos, g_line,

anim = FuncAnimation(fig, animate, frames=n, interval=100, blit=True)

plt.show()

ch = input('Save ?\n')

if ch == 'y' :
    # save the animation
    print('Saving...')
    anim.save('PSO_Original.mp4', writer = 'ffmpeg', fps = 30)
    anim.save('PSO_Slow_Mo.mp4', writer = 'ffmpeg', fps = 5)
    print('Done')

