from root_finder import find_root
import numpy as np
import matplotlib.pyplot as plt

def tridiagonal_matrix(mx, main_diagonal, near_diagonal):
    """Create a matrix of shape (mx+1, mx+1) with values on the main diagonal and 1 cell above and below the main diagonal and zeros elsewhere

    Args:
        mx (int): Size of matrix
        main_diagonal (float): Value to put on the main diagonal
        near_diagonal (float): Value to put above and below the main diagonal

    Returns:
        array: Array with main_diagonal on the main diagonal and near_diagonal above and below
    """
    # Initialise array with zeros
    arr = np.zeros((mx+1, mx+1))

    # Create selector with 1s on the main diagonal and 1 above and below it
    selector = np.tri(mx+1, mx+1, 1) - np.tri(mx+1,mx+1, -2)

    # Use selector to put near_diagonal 
    arr[selector.astype(bool)] = near_diagonal

    # Fill main diagonal with main_diagonal in-place
    np.fill_diagonal(arr, main_diagonal)
    return arr

# kwargs is used for compatibility with Crank-Nicholson
def forward_euler_step(u_j, lmbda, mx, **kwargs):
    """Do a single step using the Forward Euler scheme

    Args:
        u_j (array): Values at t0
        lmbda (float): Lambda value to use
        mx (int): Number of points in space

    Returns:
        array: Values at t1
    """

    # Initialise as zeros
    u_jp1 = np.zeros((mx+1))
    # Update everything except far left and right values
    u_jp1[1:mx] = u_j[1:mx] + lmbda*(u_j[0:mx-1] - 2*u_j[1:mx] + u_j[2:mx+1])

    return u_jp1

def backward_euler_step(u_j, lmbda, mx, solver, **kwargs):
    """Do a single step using the Backwards Euler scheme

    Args:
        u_j (array): Values at t0
        lmbda (float): Lambda value to use
        mx (int): Number of points in space
        solver (function): Solver to use

    Returns:
        array: Values at t1
    """
    # Generate tridiagonal matrix
    tridiagonal = tridiagonal_matrix(mx, 1 + 2*lmbda, -lmbda)
    # Create anonymous function and pass it to solve
    u_jp1 = solver(lambda u_jp1: np.matmul(tridiagonal, u_jp1) - u_j, u_j)

    return u_jp1

def crank_nicholson_step(u_j, lmbda, mx, solver, **kwargs):
    """Do a single step using the Crank-Nicholson scheme

    Args:
        u_j (array): Values at t0
        lmbda (float): Lambda value to use
        mx (int): Number of points in space
        solver (function): Solver to use

    Returns:
        array: Values at t1
    """
    # Generate tridiagonal matricies
    acn = tridiagonal_matrix(mx, 1 + lmbda, -lmbda/2)
    bcn = tridiagonal_matrix(mx, 1 - lmbda, lmbda/2)

    # Create anonymous function and pass it to solve
    u_jp1 = solver(lambda u_jp1: np.matmul(acn, u_jp1) - np.matmul(bcn, u_j), u_j)

    return u_jp1

def solve_pde(mx, mt, L, T, initial_function, kappa, boundary_condition, pde_step_method="forwardEuler", root_finder=find_root):
    """Solve pde with initial_conditions given by intitial_functions(x) and boundary conditions given by boundary_conditions(x, t)

    Args:
        mx (int): Number of points in space
        mt (int): Number of points in time
        L (float): Maximum x value
        T (float): Maximum t value
        initial_function (function): Function to give initial conditions
        kappa (float): Kappa value to use when solving
        boundary_condition (function): Function to give boundary conditions.  Must have signature f(x, t) -> float
        pde_step_method (str, optional): Method of 1 pde_step. Defaults to "forwardEuler".
        root_finder (function, optional): Root finder to use if required by pde_step_method. Defaults to find_root.

    Returns:
        array: Final values at t = T
    """

    # Clean inputs
    if type(pde_step_method) == str:
        if pde_step_method not in methods.keys():
            method_options = list(methods.keys())
            raise ValueError(f"{pde_step_method} is not recognised, must be one of {method_options}")
        else:
            pde_step = methods[pde_step_method]
    else:
        pde_step = pde_step_method
    
    for var, name in zip([mx, mt], ["mx", "mt"]):
        if not type(var) == int:
            raise TypeError(f"{name} must be an integer")

    for var, name in zip([L, T, kappa], ["L", "T", "kappa"]):
        if not np.can_cast(var, float):
            raise TypeError(f"{name} must be a float")
    
    # Construct boundary_function
    if np.can_cast(np.array(boundary_condition), float) and len(boundary_condition) == 2:
        boundary_function = lambda x, t: (L-x)*boundary_condition[0]+x*boundary_condition[1]
    else:
        boundary_function = boundary_condition

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
    for t in range(0, mt):
        # Forward Euler timestep at inner mesh points
        # PDE discretised at position x[i], time t[j]
        u_j = pde_step(u_j, lmbda, mx, solver=root_finder)

        # Apply boundary conditions
        u_j[0] = boundary_function(0, t)
        u_j[mx] = boundary_function(L, t)

    return u_j

# Methods available to use
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

    # Solve for each method
    forwardEuler = solve_pde(mx, mt, L, T, u_I, kappa, [0, 0], "forwardEuler", find_root)
    backwardEuler = solve_pde(mx, mt, L, T, u_I, kappa, lambda x, t: 0, "backwardEuler", find_root)
    crankNicholson = solve_pde(mx, mt, L, T, u_I, kappa, [0, 0], "crankNicholson", find_root)


    # Plot the final result and exact solution
    x = np.linspace(0, L, mx+1)
    plt.plot(x,forwardEuler, 'o', label='Forward Euler')
    plt.plot(x,backwardEuler, 'o', label='Backward Euler')
    plt.plot(x,crankNicholson, 'o', label='Crank Nicholson')

    xx = np.linspace(0,L,250)
    plt.plot(xx,u_exact(xx,T),'b-',label='exact')

    plt.title("Comparison of Forward Euler, Backward Euler and\nCrank-Nicholson step functions")

    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.legend(loc='lower center')
    plt.show()

if __name__ == "__main__":
    main()


