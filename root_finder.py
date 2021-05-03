import numpy as np

class JacobianNotConvergedError(Exception):
    pass

def jacobian_matrix(f, x, eps=1e-3, min_eps=1e-13, allclose_args=dict(rtol=1e-12)):
    """Numerically approximate jacobian matrix for f starting from eps reducing by 1 order of magnitude each time until a stable solution is found

    Args:
        f (function): (Multivariate) function
        x (list): Points to calculate jacobian
        eps (float, optional): Range of f to use for calculating jacobian. Defaults to 1e-3.

    Returns:
        np.ndarray: Jacobian matrix approximation for f at x
    """
    j_old = np.full([len(x), len(x)], fill_value=np.inf)
    j = np.zeros([len(x), len(x)], dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        while eps > min_eps and not np.allclose(j_old, j, **allclose_args):
            j_old = j.copy()
            j = np.zeros([len(x), len(x)], dtype=np.float64)
            for i, _ in enumerate(x):
                x1 = x.copy()
                x2 = x.copy()

                x1[i] += eps
                x2[i] -= eps

                f1 = np.array(f(x1))
                f2 = np.array(f(x2))

                j[:,i] = (f1 - f2) / (2 * eps)
            eps /= 10
            
    if not np.allclose(j_old, j, **allclose_args):
        raise JacobianNotConvergedError("The jacobian matrix did not converge, try scipy.optimize.fsolve as the solver")

    return j

def newton_step(f, u):
    """Do a single newton step

    Args:
        f (function): Function to do step on
        u (float): Current guess

    Returns:
        float: Improved estimate for solution to f(u) = 0
    """
    jacobian = jacobian_matrix(f, u)
    # Ensure jacobian is not singular and calculate the inverse
    inverse = np.linalg.inv(jacobian)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        return u - np.matmul(inverse, f(u))

def find_root(f, u):
    """Calculates x where f(x) = 0

    Args:
        f (function): Function to solve
        u (float): Initial guess for x

    Returns:
        float: Approximate value for x
    """
    # Clean initial guess to numpy array
    u = np.array(u, dtype=np.float64).reshape(-1)

    u_old = np.ones(u.shape) * np.inf

    while not np.allclose(u, u_old):
        u_old = u
        u = newton_step(f, u)
        if not np.isfinite(u).all():
            raise ArithmeticError
    assert np.allclose(f(u), 0), "Failed to converge"
    
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
    roots = find_root(g, np.array([0.5, 0.5, 22]))
    print(f"{roots=}")
    print(f"{g(roots)=}")

    # Compare my solution to fsolve
    from scipy.optimize import fsolve
    print(f"{fsolve(g, np.array([0.5, 0.5, 22]))=}")



if __name__ == "__main__":
    from ode_solver import solve_ode

    main()