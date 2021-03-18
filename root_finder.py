import numpy as np


def fdash(f, x, epsilon=1e-10):
    """Get approximate value for f'(x) using f(x+-epsilon)

    Args:
        f (function): Function to differentiate
        x (float): Point to integrate about
        epsilon (float, optional): Small offset to calculate differential. Defaults to 1e-10.

    Returns:
        float: f'(x)
    """
    x1 = x - epsilon
    x2 = x + epsilon

    y1 = f(x1)
    y2 = f(x2)

    return (y2 - y1) / (2*epsilon)

def newton_step(f, u):
    """Do a single newton step

    Args:
        f (function): Function to do step on
        u (float): Current guess

    Returns:
        float: Improved estimate for solution to f(u) = 0
    """
    x0 = u
    x1 = x0 - f(x0)/fdash(f, x0)
    return x1

def find_root(f, u):
    """Calculates x where f(x) = 0

    Args:
        f (function): Function to solve
        u (float): Initial guess for x

    Returns:
        float: Approximate value for x
    """
    u_old = np.inf
    while not np.isclose(u, u_old):
        u_old = u
        u = newton_step(f, u)
    return u

def main():
    f = lambda x: x ** 2 - 2 * x - 3
    roots = find_root(f, 0)
    print(f"{roots=}")
    print(f"{f(roots)=}")



if __name__ == "__main__":
    main()