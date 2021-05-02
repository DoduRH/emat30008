import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def tridiagonal_matrix(mx, main_diagonal, near_diagonal):
    arr = np.zeros((mx+1, mx+1))
    selector = np.tri(mx+1, mx+1, 1) - np.tri(mx+1,mx+1, -2)

    arr[selector.astype(bool)] = near_diagonal

    np.fill_diagonal(arr, main_diagonal)
    return arr

def forward_euler_step(u_j, lmbda, mx):
    u_jp1 = np.zeros((mx+1))      # u at next time step
    u_jp1[1:mx] = u_j[1:mx] + lmbda*(u_j[0:mx-1] - 2*u_j[1:mx] + u_j[2:mx+1])

    # Boundary conditions
    u_jp1[0] = 0
    u_jp1[mx] = 0
    return u_jp1

def backward_euler_step(u_j, lmbda, mx):
    diagonal = tridiagonal_matrix(mx, 1 + 2*lmbda, -lmbda)
    u_jp1 = fsolve(lambda u_jp1: np.matmul(diagonal, u_jp1) - u_j, u_j)

    # Boundary conditions
    u_jp1[0] = 0
    u_jp1[mx] = 0

    return u_jp1

def crank_nicholson_step(u_j, lmbda, mx):
    acn = tridiagonal_matrix(mx, 1 + lmbda, -lmbda/2)
    bcn = tridiagonal_matrix(mx, 1 - lmbda, lmbda/2)

    u_jp1 = fsolve(lambda u_jp1: np.matmul(acn, u_jp1) - np.matmul(bcn, u_j), u_j)
    
    # Boundary conditions
    u_jp1[0] = 0
    u_jp1[mx] = 0
    return u_jp1

def solve_pde(mx, mt, L, T, initial_function, kappa, pde_step_method="forwardEuler", plot=False, u_exact=None):
    # TODO: Docstring
    # TODO: Clean other inputs
    # Clean inputs
    if type(pde_step_method) == str:
        if pde_step_method not in methods.keys():
            method_options = list(methods.keys())
            raise ValueError(f"{pde_step_method} is not recognised, must be one of {method_options}")
        else:
            pde_step = methods[pde_step_method]
    else:
        pde_step = pde_step_method


    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number

    # Set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step

    # Set initial condition
    u_j = initial_function(x)

    # Solve the PDE: loop over all time points
    for _ in range(0, mt):
        # Forward Euler timestep at inner mesh points
        # PDE discretised at position x[i], time t[j]
        u_j = pde_step(u_j, lmbda, mx)
    
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


methods = {
    "forwardEuler": forward_euler_step,
    "backwardEuler": backward_euler_step,
    "crankNicholson": crank_nicholson_step,
}

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

    forwardEuler = solve_pde(mx, mt, L, T, u_I, kappa, "forwardEuler")
    backwardEuler = solve_pde(mx, mt, L, T, u_I, kappa, "backwardEuler")
    crankNicholson = solve_pde(mx, mt, L, T, u_I, kappa, "crankNicholson")


    # Plot the final result and exact solution
    x = np.linspace(0, L, mx+1)
    plt.plot(x,forwardEuler, 'o', label='Forward Euler')
    plt.plot(x,backwardEuler, 'o', label='Backward Euler')
    plt.plot(x,crankNicholson, 'o', label='Crank Nicholson')

    xx = np.linspace(0,L,250)
    plt.plot(xx,u_exact(xx,T),'b-',label='exact')

    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()


