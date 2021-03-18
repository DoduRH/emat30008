# %%
import numpy as np
from ode_solver import solve_ode
from scipy.optimize import fsolve

# %%
if __name__=="__main__":
    #G(U0) = u0 - F(u0,T)
    alpha = 1
    delta = 0.1
    beta = 0.2

    U = np.array([
        0.25,
        0.25,
    ])

    funcs = [
        lambda t, x, y: x * (1 - x) - (alpha * x * y) / (delta + x), # x
        lambda t, x, y: beta * y * (1 - (y/x)), # y
    ]

    g = lambda u: u - solve_ode(funcs, u, [0,10], 0.1, "RK4")[-1]

    guess = [0.5, 0.5]

    res = fsolve(g, guess)
    print(res)
    print(g(res))
    print(np.isclose(g(res), [0,0]))

# %%
