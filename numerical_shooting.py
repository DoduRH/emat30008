# %%
import numpy as np
from ode_solver import solve_ode
from repeat_finder import find_repeats
from scipy.optimize import fsolve

def shoot(f, initial, tmax=200):
    """Use numerical shooting to calculate the initial conditions and period for f

    Args:
        f (list): system of ODEs as a list
        initial (list): Initial conditions to start from
        tmax (int, optional): Max value to get initial period guess. Defaults to 200.

    Returns:
        list: List containing initial conditions and period of the function
    """

    # Find approximate location of periodic behaviour
    # TODO: Steadily increment tmax until full period is found?
    rk4_solution = solve_ode(f, initial, np.linspace(0, tmax, 500), 0.1, "rk4")
    initial_conditions, period = find_repeats(rk4_solution, abs_tol=0.08)

    # Setup g
    g = lambda U: [
        *(U[:-1] - solve_ode(f, U[:-1], [0, U[-1]], 0.1, "RK4")[-1]),
        f[0](U[-1], *U[:-1]), # dx/dt(0) = 0
    ]

    # Find roots of g
    # TODO: Option to use custom root finder
    orbit = fsolve(g, [*initial_conditions, period])

    # Make sure only singular orbit was found by dividing the period 
    # period until it no longer gives g(orbit) approx= 0
    x0 = orbit[:-1]
    divisor = 2
    
    # This assumes the minimum period is 1
    while divisor < orbit[-1]: 
        while np.allclose(orbit[:-1], x0, atol=1e-3):
            orbit[-1] /= divisor
            x0 = solve_ode(f, orbit[:-1], [0, orbit[-1]], 0.1, "RK4")[-1]
        orbit[-1] *= divisor # revert final division
        x0 = orbit[:-1] # Reset x0
        divisor += 1
        
    # Run fsolve once more to re-align to a single orbit
    # TODO: Option to use custom root finder
    orbit = fsolve(g, orbit)

    return orbit
# %%
if __name__=="__main__":
    # Setup equations
    # Lokta Volterra variables
    alpha = 1
    delta = 0.1
    beta = 0.2

    # Lokta-Volterra equations
    funcs = [
        lambda t, x, y: x * (1 - x) - (alpha * x * y) / (delta + x), # dx/dt
        lambda t, x, y: beta * y * (1 - (y/x)), # dy/dt
    ]

    # Lokta-Voltera initial conditions
    initial = [
        0.25,
        0.25,
    ]

    print(f'{shoot(funcs, initial)=}')

# %%
