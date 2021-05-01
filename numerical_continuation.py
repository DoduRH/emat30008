from numerical_shooting import shoot
from root_finder import find_root
import numpy as np

def continuation(myode, x0, par0, vary_par, par_max, hmax=0.1, discretisation=shoot, solver=find_root):
    """finds the line

    Args:
        myode (function): Function to find line for
        x0 (List): Initial conditions
        par0 (Dict): Initial Parameters
        vary_par (str, optional): Name of parameter to vary
        vary_par_vals (itterable, optional): Values to assign to par0[vary_par]. Defaults to np.linspace(0, 1, 100).
        discretisation (function, optional): Numerical shooting method. Defaults to shoot.
        solver (function, optional): Root finder to use. Defaults to find_root.
    
    Returns:
        (array): Array of points 
    """
    output = []
    last_result = x0.copy()
    while par0[vary_par] + hmax < par_max:
        par0[vary_par] += hmax
        last_result = solver(discretisation(myode), last_result, args=par0)
        print(f"Done {vary_par=} {par0[vary_par]=} {last_result=} {myode(last_result, par0)=}")
        output.append((par0[vary_par], *last_result))
    
    par0[vary_par] = par_max
    last_result = fsolve(myode, last_result, args=par0)
    output.append((par0[vary_par], *last_result))
    print(f"Done {vary_par=} {par0[vary_par]=} {last_result=} {myode(last_result, par0)=}")

    return output

if __name__ == "__main__":
    par0 = dict(
        beta = 0
    )

    x0 = [0.2, 0.2]

    # Hopf Bifurcation
    theODE = lambda t, U, p: [
                p['beta'] * U[0] - U[1] - U[0] * (U[0] ** 2 + U[1] ** 2),
        U[0] +  p['beta'] * U[1] -        U[1] * (U[0] ** 2 + U[1] ** 2),
    ]

    par0 = dict(
        alpha = 1,
        delta = 0.1,
        beta = 0.2,
    )

    def Lokta_Volterra(t, U, p):
        x = U[0] * (1 - U[0]) - (p['alpha'] * U[0] * U[1]) / (p['delta'] + U[0]) # dx/dt
        y = p['beta'] * U[1] * (1 - (U[1]/U[0])) # dy/dt

        return [x, y]

    # Lokta-Voltera initial conditions
    initial = [
        1.5,
    ]

    from scipy.optimize import fsolve

    par0 = dict(c=-2)
    theODE = lambda U, p: [
        U[0] ** 3 - U[0] + p['c']
    ]

    #par0 = dict(alpha=0)
    #theODE = lambda t, r, p: p['alpha'] * r + r ** 3 - r ** 5

    results = continuation(
        theODE,  # the ODE to use
        initial,  # the initial state
        par0,  # the initial parameters
        vary_par="c",  # the parameter to vary
        par_max=2,
        hmax=0.05, # max step size
        discretisation=lambda x: x,  # the discretisation to use
        solver=fsolve,  # the solver to use
    )
    
    import matplotlib.pyplot as plt
    import numpy as np

    r = np.array(results).T
    x = r[0]
    ys = r[1]

    plt.xlabel("Beta")

    plt.plot(x, ys)

    plt.legend(["Stable Y", "Stable X", "Time"])

    plt.show()
