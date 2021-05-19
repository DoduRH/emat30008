# %%
# Imports
from ode_solver import solve_ode
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import nan, exp

# %%
# Setup ODE and analytical functions
func = lambda t, x: x
analytic_sol = lambda t, x: exp(t)
t = [0, 1]
x0 = 1

# %%
# Calculate error vs hmax
h_size = np.logspace(-9, 0, 101)
analytic = analytic_sol(t[-1], nan)
results = {}
methods = ['Euler', 'RK4']

# Run for each method
for method in methods:
    results[method] = []
    start = time.time()
    # Loop over each h size and show progress bar
    for h in tqdm(h_size, desc=method):
        r = solve_ode(func, x0, t, h, method)
        results[method].append(abs(r[-1] - analytic))
    end = time.time()

# %%
# Plot error against hmax on a log-log scale
for data in methods:
    plt.loglog(h_size, results[data])

plt.xlabel("HMAX")
plt.ylabel("Error")

plt.legend(methods)

plt.show()

# %%
# Calculate h values to give equal errors
target_error = 1e-6
euler_h = h_size[(np.abs(np.array(results['Euler']) - target_error)).argmin()]
rk4_h = h_size[(np.abs(np.array(results['RK4']) - target_error)).argmin()]

print(f"{euler_h=}")
print(f"{rk4_h=}")

# %%
# Plot error against hmax on a log-log scale with lines showing error bars
for data in methods:
    plt.loglog(h_size, results[data])

plt.axhline(target_error)

plt.vlines([euler_h, rk4_h], ymin=0, ymax=target_error)


plt.xlabel("HMAX")
plt.ylabel("Error")

plt.legend(methods)

plt.show()
# %%