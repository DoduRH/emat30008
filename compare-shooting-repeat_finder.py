# %%
# Imports
from ode_solver import solve_ode
from numerical_shooting import shoot
from repeat_finder import find_period
import numpy as np
import matplotlib.pyplot as plt

# %%
# Setup equations
# Predator-prey variables
alpha = 1
delta = 0.1
beta = 0.2

# Predator-prey equations
funcs = lambda t, U: [
    U[0] * (1 - U[0]) - (alpha * U[0] * U[1]) / (delta + U[0]), # dU[0]/dt
    beta * U[1] * (1 - (U[1]/U[0])), # dU[1]/dt
]

# Predator-prey initial conditions
initial = [
    0.25,
    0.25,
]

shootingResults = shoot(funcs, initial)
t = np.linspace(0, shootingResults[-1], 100)
shootingSolution = solve_ode(funcs, shootingResults[:-1], t, 0.1, "RK4")

print(f'Shooting starting position {shootingResults[0]}, {shootingResults[1]} with period {shootingResults[2]}.  It has an error of {np.linalg.norm(shootingSolution[0] - shootingSolution[-1])}')

findPeriodResults = find_period(lambda t: solve_ode(funcs, initial, t, 0.1, "rk4"))
t = np.linspace(0, findPeriodResults[-1], 100)
findPeriodSolution = solve_ode(funcs, findPeriodResults[:-1], t, 0.1, "RK4")

print(f'Find_period starting position {findPeriodResults[0]}, {findPeriodResults[1]} with period {findPeriodResults[2]}. It has an error of {np.linalg.norm(findPeriodSolution[0] - findPeriodSolution[-1])}')


# %%
# Plot 1 period to check it is correct
# Plot find_period results

x_vals, y_vals = findPeriodSolution.T

plt.plot(x_vals, y_vals, label="Find Period")

# Plot shooting results

x_vals, y_vals = shootingSolution.T

plt.plot(x_vals, y_vals, label="Shooting")
plt.scatter(*findPeriodResults[:-1], label="Find Period")
plt.scatter(*shootingResults[:-1], label="Shooting")
plt.legend()
plt.show()
