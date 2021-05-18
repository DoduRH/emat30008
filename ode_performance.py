# %%
# Import libraries
import time
from ode_solver import solve_ode
from math import nan, exp

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

#h_values = {
#    "euler": 0.1,
#    "rk4": 0.1
#}
# %%
# Run and measure execution time for each method
# Number of seconds to run for
num_seconds = 30

for method, h in h_values.items():
    # Time it
    # Run for 30 seconds to reduce variance
    start_time = time.time()
    total = 0
    while time.time() - start_time < num_seconds:
        res = solve_ode(func, x0, t, h, method)
        total += 1
    end_time = time.time()
    # Print error and execution time
    print(f"Error for {method} is {abs(res[-1][0] - analytic_sol(t[-1], nan))} with a step size of {h}")
    print(f"and ran {total} itterations in {round(end_time - start_time, 3)} seconds giving {round(total/(end_time-start_time), 3)} iterations/second\n")
