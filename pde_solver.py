import numpy as np
import matplotlib.pyplot as plt

def forward_euler_step(u_j, lmbda, mx):
    u_jp1 = np.zeros((mx+1))      # u at next time step
    u_jp1[1:mx] = u_j[1:mx] + lmbda*(u_j[0:mx-1] - 2*u_j[1:mx] + u_j[2:mx+1])

    # Boundary conditions
    u_jp1[0] = 0
    u_jp1[mx] = 0
    return u_jp1

def solve_pde(mx, mt, L, T, initial_function, kappa, plot=False, u_exact=None):
    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number

    # Set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step

    # Set initial condition
    u_j = initial_function(x)

    # Solve the PDE: loop over all time points
    for _ in range(0, mt):
        # Forward Euler timestep at inner mesh points
        # PDE discretised at position x[i], time t[j]
        u_j = forward_euler_step(u_j, lmbda, mx)
    
    if plot:
        plot_pde(x, u_j, u_exact, T, L)

    return u_j

def plot_pde(x, u_j, u_exact=None, T=None, L=None):
    # Plot the final result and exact solution
    plt.plot(x,u_j,'ro',label='num')

    if u_exact is not None and T is not None and L is not None:
        xx = np.linspace(0,L,250)
        plt.plot(xx,u_exact(xx,T),'b-',label='exact')

    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.legend(loc='upper right')
    plt.show()

def main():
    from math import pi
    # Set problem parameters/functions
    kappa = 1.0   # diffusion constant
    L=1.0         # length of spatial domain
    T=0.5         # total time to solve for
    def u_I(x):
        # initial temperature distribution
        y = np.sin(pi*x/L)
        return y

    def u_exact(x,t):
        # the exact solution
        y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
        return y

    # Set numerical parameters
    mx = 10     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time

    u_j = solve_pde(mx, mt, L, T, u_I, kappa, True, u_exact=u_exact)

if __name__ == "__main__":
    main()


