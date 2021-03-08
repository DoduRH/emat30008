# %%
# import
from ode_solver import solve_ode
from scipy.integrate import odeint

# %%
# Setup system of equations
funcs = [
    lambda t, x, y: y, # x
    lambda t, x, y: -x, # y
]

initial = [
    1,
    1,
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
print(solve_ode(funcs, initial, [0, 1], 1e-5, "rk4"))
print(odeint(g, initial, [0, 1], hmax=0.1))
