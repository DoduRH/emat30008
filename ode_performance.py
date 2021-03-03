# %%
# Import libraries
from timer import timer
from ode_solver import solve_ode
from math import nan, exp
from numba import njit

# %%
# Set variables
func = lambda t, x: x
analytic_sol = lambda t, x: exp(t)
t = [0, 1]
x0 = 1
h_values = {
    "euler": 1e-6,
    "rk4": 1e-1
}
# %%
# Run and measure execution time for each method
for method, h in h_values.items():
    # Setup func as a function that can be compiled
    f = njit(func)

    # Run solve_ode once to compile functions
    res = solve_ode(f, x0, t, h, method)

    # Time it
    with timer() as clock:
        # Run 100 times to reduce variance
        for i in range(100):
            res = solve_ode(f, x0, t, h, method)
        # Get elapsed time
        time_spent = clock.elapse
        # Print error and execution time
        print(f"Error for {method} is {abs(res[-1] - analytic_sol(t[-1], nan))} and took {time_spent} seconds with a step size of {h}")
