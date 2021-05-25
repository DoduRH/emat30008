# %%
# Import libraries
from ode_solver import solve_ode
from math import nan, exp
from measure_performance import perf_measure

# %%
# Set variables
func = lambda t, x: x
analytic_sol = lambda t, x: exp(t)
t = [0, 1]
x0 = 1

measurements = [
    dict(method="rk4", h=0.1),
    dict(method="euler", h=0.1),
    dict(method="euler", h=1e-6),
]

# %%
# Run and measure execution time for each method

for test in measurements:
    # Get values from dictionary
    method = test['method']
    h = test['h']
    # Time it
    # Run for 30 seconds to reduce variance
    total_time, total_iterations, variance, res = perf_measure(solve_ode, 30, func, x0, t, h, method)

    # Print error and execution time
    print(f"Error for {method} is {abs(res[-1][0] - analytic_sol(t[-1], nan))} with a step size of {h}")
    print(f"and ran {total_iterations} itterations in {round(total_time, 3)} seconds with a variance of {variance} giving {round(total_iterations/total_time, 3)} iterations/second\n")
