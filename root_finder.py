import numpy as np

class JacobianNotConvergedError(Exception):
    pass



def jacobian_matrix(f, x, args=None, eps=1e-8):
    """Numerically approximate jacobian matrix for f starting from eps reducing by 1 order of magnitude each time until a stable solution is found

    Args:
        f (function): (Multivariate) function
        x (list): Points to calculate jacobian
        args (list, optional): Arguments to pass to f. Defaults to no arguments.
        eps (float, optional): Range of f to use for calculating jacobian. Defaults to 1e-3.

    Returns:
        np.ndarray: Jacobian matrix approximation for f at x
    """

    if args is None:
        args = []

    with np.errstate(divide='ignore', invalid='ignore'):
        j = np.zeros([len(x), len(x)], dtype=np.float64)
        for i, _ in enumerate(x):
            x0 = x.copy()
            x1 = x.copy()
            x2 = x.copy()
            x3 = x.copy()

            x0[i] += 2 * eps
            x1[i] += eps
            x2[i] -= eps
            x3[i] -= 2 * eps

            f0 = np.array(f(x0, *args))
            f1 = np.array(f(x1, *args))
            f2 = np.array(f(x2, *args))
            f3 = np.array(f(x3, *args))

            j[:,i] = (-f0 + 8 * f1 - 8 * f2 + f3) / (12 * eps)
            
    if not np.isfinite(j).all():
        raise JacobianNotConvergedError("The jacobian matrix did not converge, try scipy.optimize.fsolve as the solver")

    return j

def newton_step(f, u, args):
    """Do a single newton step

    Args:
        f (function): Function to do step on
        u (float): Current guess

    Returns:
        float: Improved estimate for solution to f(u) = 0
    """
    jacobian = jacobian_matrix(f, u, args)
    # Ensure jacobian is not singular and calculate the inverse
    inverse = np.linalg.inv(jacobian)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        return u - np.matmul(inverse, f(u, *args))

def find_root(f, u, args=None):
    """Calculates x where f(x) = 0

    Args:
        f (function): Function to solve
        u (float): Initial guess for x

    Returns:
        float: Approximate value for x
    """
    # Clean initial guess to numpy array
    u = np.array(u, dtype=np.float64).reshape(-1)
    if args is None:
        args = []

    if type(args) != list:
        args = [args]

    u_old = np.ones(u.shape) * np.inf

    while not np.allclose(u, u_old, atol=1e-12):
        u_old = u
        u = newton_step(f, u, args)
        if not np.isfinite(u).all():
            raise ArithmeticError
    assert np.allclose(f(u, *args), 0), "Failed to converge"
    
    return u

def main():
    beta = 0.5
    sigma = -1

    f = lambda t, U: [
        beta * U[0] -        U[1] + sigma * U[0] * (U[0]**2 + U[1]**2),
               U[0] + beta * U[1] + sigma * U[1] * (U[0]**2 + U[1]**2),
    ]

    g = lambda U: [
        *(U[:-1] - solve_ode(f, U[:-1], [0, U[-1]], 0.1, "RK4")[-1]),
        f(U[-1], U[:-1])[0], # dx/dt(0) = 0
    ]

    approx_period = np.array([1, 0, 7])

    # Find roots of g
    orbit = find_root(g, approx_period)
    print(orbit)
    
    # Scalar root find test
    print(f"{find_root(lambda x: x ** 2 - 2 * x - 3, 4)=}")

    # Setup preadator prey equation
    alpha = 1
    delta = 0.1
    beta = 0.2

    funcs = lambda t, U: [
        U[0] * (1 - U[0]) - (alpha * U[0] * U[1]) / (delta + U[0]), # dU[0]/dt
        beta * U[1] * (1 - (U[1]/U[0])), # dU[1]/dt
    ]

    g = lambda U: [
        *(U[:-1] - solve_ode(funcs, U[:-1], [0, U[-1]], 0.1, "RK4")[-1]),
        U[0] * (1 - U[0]) - (alpha * U[0] * U[1]) / (delta + U[0]), # dx/dt(0) = 0
    ]

    # Run find roots with prediction of x = 0.5, y = 0.5 and T = 22
    find_root_roots = find_root(g, np.array([0.5, 0.5, 22]))
    print(f"{find_root_roots=}")
    print(f"{g(find_root_roots)=}")

    # Compare my solution to fsolve
    from scipy.optimize import fsolve
    fsolve_roots = fsolve(g, np.array([0.5, 0.5, 22]))
    print(f"{fsolve_roots=}")
    print(f"{g(fsolve_roots)=}")



if __name__ == "__main__":
    from ode_solver import solve_ode

    main()