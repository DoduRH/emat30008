import numpy as np
from collections.abc import Iterable


def fdash(f, x, pos=0, epsilon=1e-10):
    """Get approximate value for f'(x) using f(x+-epsilon)

    Args:
        f (function): Function to differentiate
        x (float): Point to integrate about
        epsilon (float, optional): Small offset to calculate differential. Defaults to 1e-10.

    Returns:
        float: f'(x)
    """
    x1 = x.copy()
    x2 = x.copy()

    x1[pos] = x1[pos] - epsilon
    x2[pos] = x2[pos] + epsilon

    y1 = f(*x1)
    y2 = f(*x2)

    return (y2 - y1) / (2*epsilon)

def jacobian_matrix(f, x, eps=1e-10):
    J = np.zeros([len(x), len(x)], dtype=np.float32)

    for i, _ in enumerate(x):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += eps
        x2[i] -= eps

        f1 = f(x1)
        f2 = f(x2)

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
    return u - np.matmul(np.linalg.inv(jacobian), f(u))

def find_root(f, u):
    """Calculates x where f(x) = 0

    Args:
        f (function): Function to solve
        u (float): Initial guess for x

    Returns:
        float: Approximate value for x
    """
    if not isinstance(u, Iterable):
        u = np.array([u], dtype=np.float32)

    elif type(u) != np.ndarray:
        u = np.array(u, dtype=np.float32)

    u_old = np.ones(u.shape) * np.inf

    while not np.allclose(u, u_old):
        u_old = u
        u = newton_step(f, u)
    return u

def main():
    # Setup preadator prey equation
    alpha = 1
    delta = 0.1
    beta = 0.2

    funcs = [
        lambda t, x, y: x * (1 - x) - (alpha * x * y) / (delta + x), # dx/dt
        lambda t, x, y: beta * y * (1 - (y/x)), # dy/dt
    ]

    g = lambda U: np.array([
        *(U[:-1] - solve_ode(funcs, U[:-1], [0, U[-1]], 0.1, "RK4")[-1]),
        U[0] * (1 - U[0]) - (alpha * U[0] * U[1]) / (delta + U[0]), # dx/dt(0) = 0
    ])

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