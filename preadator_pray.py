# %%
# import
from ode_solver import solve_ode
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# %%
# Setup system of equations
alpha = 1
delta = 0.1
beta = 0.2

funcs = [
    lambda t, x, y: x * (1 - x) - (alpha * x * y) / (delta + x), # dx/dt
    lambda t, x, y: beta * y * (1 - (y/x)), # dy/dt
]

initial = [
    0.25,
    0.25,
]

t = np.linspace(0, 200, 500)

stepsize = 0.1

# %%
# Get solutions for RK4 and euler
rk4_solution = np.array(solve_ode(funcs, initial, t, stepsize, "rk4"))

sols = {
    "rk4 solution x": rk4_solution[:,0],
    "rk4 solution y": rk4_solution[:,1],
}

phase_sols = {
    "rk4 solution": (rk4_solution[:,0], rk4_solution[:,1]),
}

# %%
from repeat_finder import find_repeats

print(find_repeats(rk4_solution, abs_tol=0.07))

guess = [0.5, 0.5]

g = lambda U: [
    *(U[:-1] - solve_ode(funcs, U[:-1], [0, U[-1]], 0.1, "RK4")[-1]),
    U[0] * (1 - U[0]) - (alpha * U[0] * U[1]) / (delta + U[0]), # dx/dt(0) = 0
]

repeats = fsolve(g, np.array([*guess, 22]))

#repeats, period = find_repeats(rk4_solution, 0.01)
print(f'Repeats found at {repeats[0]} and {repeats[1]} with period of {repeats[2]}')

# %%
# Plot RK4, Euler and analytic solutions
labels = []
colours = [u'#1f77b4', u'#ff7f0e']

# %%
if False:
    plt.xlabel("Time (t)")
    for repeat, (label, y), colour in zip(repeats, sols.items(), colours):
        plt.plot(t, y, colour)
        labels.append(label)

        plt.hlines(repeat, np.amin(t), np.amax(t), colour)

    plt.ylabel("y")

    plt.legend(labels)

    plt.show()

# %%
labels = []
for label, (x, y) in phase_sols.items():
    plt.plot(x, y)
    plt.xlabel("x")
    labels.append(label)

plt.scatter(*repeats[:2])

plt.ylabel("y")

plt.legend(labels)

plt.show()


# %%
