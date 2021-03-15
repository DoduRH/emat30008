# %%
# import
from ode_solver import solve_ode
import numpy as np
import matplotlib.pyplot as plt

# %%
# Setup system of equations
alpha = 1
delta = 0.1
beta = 0.2

funcs = [
    lambda t, x, y: x * (1 - x) - (alpha * x * y) / (delta + x), # x
    lambda t, x, y: beta * y * (1 - (y/x)), # y
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

repeats = find_repeats(rk4_solution, 0.01)
print(f'Repeats found at {repeats[0]} and {repeats[1]}')

# %%
# Plot RK4, Euler and analytic solutions
labels = []
colours = [u'#1f77b4', u'#ff7f0e']

# %%
plt.xlabel("Time (t)")
for repeat, (label, y), colour in zip(repeats, sols.items(), colours):
    plt.plot(t, y, colour)
    labels.append(label)

    plt.hlines(repeat, np.amin(t), np.amax(t), colour)

plt.ylabel("y")

plt.legend(labels)

plt.show()

# %%
for label, (x, y) in phase_sols.items():
    plt.plot(x, y)
    plt.xlabel("x")
    labels.append(label)

plt.scatter(*repeats)

plt.ylabel("y")

plt.legend(labels)

plt.show()


# %%
