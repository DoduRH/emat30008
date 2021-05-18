# %%
# import
from ode_solver import solve_ode
import numpy as np
import matplotlib.pyplot as plt

# %%
# Setup system of equations
funcs = lambda t, U: [
    U[1], # x
    -U[0], # y
]

initial = [
    2,
    1,
]

t = np.linspace(0, 20, 100)

analytic_sol = [
    np.sin(t) - 2* (-np.cos(t)),
    np.cos(t) - 2* (np.sin(t)),
]

stepsize = 0.1

# %%
# Get solutions for RK4 and euler
rk4_solution = np.array(solve_ode(funcs, initial, t, stepsize, "rk4"))
euler_solution = np.array(solve_ode(funcs, initial, t, stepsize, "euler"))

sols = {
    "rk4 solution x": rk4_solution[:,1],
    "rk4 solution y": rk4_solution[:,0],
    "euler_solution x": euler_solution[:,0],
    "euler_solution y": euler_solution[:,1],
    "analytic x": analytic_sol[0],
    "analytic y": analytic_sol[1],
}

# %%
# Plot RK4, Euler and analytic solutions
labels = []
for i, (label, y) in enumerate(sols.items()):
    if "rk4" in label:
        plt.plot(t, y, linewidth=3)
    else:
        plt.plot(t, y)
    labels.append(label)

plt.xlabel("Time (t)")
plt.ylabel("y")

plt.title(f'Comparing the analytic and computed solutions to $\\ddot x = x$\nwith initial conditions x = {initial[0]}, y = {initial[1]} and step size of {stepsize} for ${min(t)} \\leq t \\leq {max(t)}$')

plt.legend(labels)

plt.savefig(f"graphs/system of ode solution {min(t)}-{max(t)}.svg")
plt.show()
# %%
