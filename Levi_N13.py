# Written by - Souritra Garai
# Contact - souritra.garai@iitgn.ac.in, sgarai65@gmail.com

# Numpy Universal Function to 
# caculate the Levi N.13 function
# used for testing optimization algorithms

import numpy as np
from numba import jit

# the x and y limits of search space for Levi N.13 function
limits = np.array([[-10, 10], [-10, 10]])

# The function definition
# refer - https://en.wikipedia.org/wiki/Test_functions_for_optimization
@jit(nopython=True)
def f(x, y) :

    return (    (np.sin(3*np.pi*x)**2) 
            +   ((x - 1)**2)*(1 + (np.sin(3*np.pi*y)**2))
            +   ((y - 1)**2)*(1 + (np.sin(2*np.pi*y)**2))   )

# For iterative optimization algorithms, 
# the Levi function will be called multiple times,
# numba is used to optimize the function for multiple calls
        
if __name__ == "__main__":

    # 3D Surface plot of the Levi N.13 function

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import time

    n = 1000
    x = np.linspace(-10, 10, n)
    y = np.linspace(-10, 10, n)

    x, y = np.meshgrid(x, y, indexing='ij')
    t = time.time()
    z = f(x, y)
    print('Time required for calculation :', time.time()-t, 's')

    fig = plt.figure()
    ax  = plt.axes(projection='3d')

    ax.plot_surface(x, y, z,cmap='magma', edgecolor='none')
    ax.set_title('Levi N.13 Function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z = f(x,y)')
    plt.show()
