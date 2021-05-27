# %%
import numpy as np
from ode_solver import solve_ode
from repeat_finder import TimePeriodNotFoundError, find_period
from root_finder import find_root

def shoot(f, initial, approximate_period=None, tmax=np.inf, solver=find_root, ODEparams=None):
    """Use numerical shooting to calculate the initial conditions and period for f

    Args:
        f (function): f, must return a list.  Is called as `f(t, U, *ODEparams)`
        initial (list): Initial conditions to start from
        approximate_period (float): Approximate period. Defaults to None in which case `find_period` will be used to estimate
        tmax (int, optional): Max value to get initial period guess. Defaults to 200.
        ODEparams (list, optional): Optional extra parameters to pass to the ODE. Defaults to []

    Returns:
        list: List containing initial conditions and period of the function
    """

    if ODEparams is None:
        ODEparams = []

    # Find approximate location of periodic behaviour
    if approximate_period is None:
        approx_period = find_period(lambda t: solve_ode(f, initial, t, 0.1, "rk4", ODEparams), tmax=tmax)
        if approx_period[-1] == -1:
            raise TimePeriodNotFoundError
    else:
        approx_period = np.append(initial, approximate_period)
    
    if approx_period[-1] < 0:
        raise ValueError(f"Invalid time period {approx_period[-1]}")

    # Setup g
    g = lambda U: [
        *(U[:-1] - solve_ode(f, U[:-1], [0, U[-1]], 0.1, "RK4", ODEparams)[-1]),
        f(U[-1], U[:-1], *ODEparams)[0], # dx/dt(0) = 0
    ]

    # Find roots of g
    orbit = solver(g, approx_period)

    return orbit
# %%
if __name__=="__main__":
    # Setup equations
    # Preadator-prey variables
    alpha = 1
    delta = 0.1
    beta = 0.2

    # Preadator-prey equations
    funcs = lambda t, U: [
        U[0] * (1 - U[0]) - (alpha * U[0] * U[1]) / (delta + U[0]), # dU[0]/dt
        beta * U[1] * (1 - (U[1]/U[0])), # dU[1]/dt
    ]

    # Preadator-prey initial conditions
    initial = [
        0.25,
        0.25,
    ]

    *periodic_conditions, period = shoot(funcs, initial)
    print(f'{periodic_conditions=} {period=}')

    # Plot the solution
    import matplotlib.pyplot as plt
    plt.scatter(*initial, label="Initial conditions")

    t = np.linspace(0, 120, 1000)
    plt.plot(*solve_ode(funcs, initial, t, 0.1, method="rk4").T, label="Non periodic motion")

    plt.scatter(*periodic_conditions, label="Periodic initial conditions")
    t = np.linspace(0, period, 100)
    plt.plot(*solve_ode(funcs, periodic_conditions, t, 0.1, method="rk4").T, label="Periodic motion")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Shooting using the preadator-prey equation")

    plt.legend()
    plt.show()
# %%
