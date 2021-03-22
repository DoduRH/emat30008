import numpy as np


def jacobian_matrix(f, x, eps=1e-10):
    """Numerically approximate jacobian matrix for f

    Args:
        f (function): (Multivariate) function
        x (list): Points to calculate jacobian
        eps (float, optional): Range of f to use for calculating jacobian. Defaults to 1e-10.

    Returns:
        np.ndarray: Jacobian matrix approximation for f at x
    """
    J = np.zeros([len(x), len(x)], dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        for i, _ in enumerate(x):
            x1 = x.copy()
            x2 = x.copy()

            x1[i] += eps
            x2[i] -= eps

            f1 = np.array(f(x1))
            f2 = np.array(f(x2))

            J[:,i] = (f1 - f2) / (2 * eps)

    return J

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
    return u

def main():
    # Scalar root find test
    print(f"{find_root(lambda x: x ** 2 - 2 * x - 3, 4)=}")

    # Setup preadator prey equation
    alpha = 1
    delta = 0.1
    beta = 0.2

    funcs = [
        lambda t, x, y: x * (1 - x) - (alpha * x * y) / (delta + x), # dx/dt
        lambda t, x, y: beta * y * (1 - (y/x)), # dy/dt
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