# %%
import numpy as np
from ode_solver import solve_ode
from repeat_finder import TimePeriodNotFoundError, find_period
from root_finder import find_root

def shoot(f, initial, approximate_solution=None, tmax=np.inf, solver=find_root, ODEparams=[]):
    """Use numerical shooting to calculate the initial conditions and period for f

    Args:
        ODE (function): ODE, must return a list.  Is called as ODE(t, U, *ODEparams)
        initial (list): Initial conditions to start from
        tmax (int, optional): Max value to get initial period guess. Defaults to 200.
        ODEparams (list, optional): Optional extra parameters to pass to the ODE. Defaults to []

    Returns:
        list: List containing initial conditions and period of the function
    """

    # Find approximate location of periodic behaviour
    if approximate_solution is None:
        approx_period = find_period(lambda t: solve_ode(f, initial, t, 0.1, "rk4", ODEparams), tmax=tmax)
    else:
        approx_period = approximate_solution

    if approx_period[-1] == -1:
        raise TimePeriodNotFoundError

    # NOTE: Is f[0](U[-1], *U[:-1]) a general phase condition?
    # Setup g
    g = lambda U: [
        *(U[:-1] - solve_ode(f, U[:-1], [0, U[-1]], 0.1, "RK4", ODEparams)[-1]),
        f(U[-1], U[:-1], *ODEparams)[0], # dx/dt(0) = 0
    ]

    # Find roots of g
    orbit = solver(g, approx_period)

    # Make sure only singular orbit was found by dividing the period 
    # period until it no longer gives g(orbit) approx= 0
    x0 = orbit[:-1]
    divisor = 2
    
    # This assumes the minimum period is 1
    if approximate_solution is None:
        while divisor < orbit[-1]: 
            while np.allclose(orbit[:-1], x0, rtol=1e-3):
                orbit[-1] /= divisor
                x0 = solve_ode(f, orbit[:-1], [0, orbit[-1]], 0.1, "RK4", ODEparams)[-1]
            orbit[-1] *= divisor # revert final division
            x0 = orbit[:-1] # Reset x0
            divisor += 1
        
    # Run solver once more to re-align to a single orbit
    orbit = solver(g, orbit)

    return orbit
# %%
if __name__=="__main__":
    # Setup equations
    # Lokta Volterra variables
    alpha = 1
    delta = 0.1
    beta = 0.2

    # Lokta-Volterra equations
    funcs = lambda t, U: [
        U[0] * (1 - U[0]) - (alpha * U[0] * U[1]) / (delta + U[0]), # dU[0]/dt
        beta * U[1] * (1 - (U[1]/U[0])), # dU[1]/dt
    ]

    # Lokta-Voltera initial conditions
    initial = [
        0.25,
        0.25,
    ]

    print(f'{shoot(funcs, initial)=}')

# %%
