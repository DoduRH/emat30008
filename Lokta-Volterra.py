# %%
from ode_solver import solve_ode
from numerical_shooting import shoot
import numpy as np
import matplotlib.pyplot as plt

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

g = lambda U: [
    *(U[:-1] - solve_ode(funcs, U[:-1], [0, U[-1]], 0.1, "RK4")[-1]),
    funcs[0](U[-1], *U[:-1]), # dx/dt(0) = 0
]

# Lokta-Voltera initial conditions
initial = [
    0.25,
    0.25,
]

x, y, T = shoot(funcs, initial)

print(f'{g([x, y, T])=}')

print(f'starting position {x}, {y} with period {T}')

# %%
# Plot 1 period to check it is correct
t = np.linspace(0, T, 100)

solution = solve_ode(funcs, [x, y], t, 0.1, "RK4")
x_vals, y_vals = solution.T

plt.plot(x_vals, y_vals)
plt.show()
