# %%
# import
from ode_solver import solve_ode
import numpy as np
import matplotlib.pyplot as plt

# %%
# Setup system of equations
funcs = [
    lambda t, x, y: y, # x
    lambda t, x, y: -x, # y
]

initial = [
    2,
    1,
]

t = np.linspace(0, 10, 50)

analytic_sol = [
    np.sin(t) - 2* (-np.cos(t)),
    np.cos(t) - 2* (np.sin(t)),
]

# %%
def funcs_wrapper(U, t, funcs):
    output = []
    for f in funcs:
        output.append(f(t, *U))
    return output

# Translate list of lambdas into function for scipy
g = lambda U, t: funcs_wrapper(U, t, funcs)

# %%
rk4_solution = np.array(solve_ode(funcs, initial, t, 0.1, "rk4"))
euler_solution = np.array(solve_ode(funcs, initial, t, 0.1, "euler"))

sols = {
    "rk4 solution x": rk4_solution[:,1],
    "rk4 solution y": rk4_solution[:,0],
    "euler_solution x": euler_solution[:,0],
    "euler_solution y": euler_solution[:,1],
    "analytic x": analytic_sol[0],
    "analytic y": analytic_sol[1],
}


labels = []
for i, (label, y) in enumerate(sols.items()):
    plt.plot(t, y)
    labels.append(label)

plt.xlabel("Time (t)")
plt.ylabel("y")

plt.title(r'Comparing the analytic and computed solutions to $\ddot{x} = x$ with initial conditions x = \mstep size of 0.1 between 0 and 10')

plt.legend(labels)

plt.savefig("graphs/system of ode solution.svg")
# %%
