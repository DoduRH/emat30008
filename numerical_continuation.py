from numerical_shooting import shoot
from root_finder import find_root
import numpy as np

def continuation(func, x0, par0, vary_par, par_max, delta_multiplier=0.1, discretisation=lambda x: x, solver=find_root):
    """Finds points where the function is stable

    Args:
        func (function): Function to find the line for
        x0 (List): Initial conditions
        par0 (Dict): Initial parameters
        vary_par (str): Key of dictionary to vary
        par_max (float): Max value of var_par
        delta_multiplier (float, optional): Jump between points. Defaults to 0.1.
        discretisation (function, optional): Discretisation function. Defaults to passing func through.
        solver (function, optional): Solver to use. Defaults to find_root.

    Returns:
        array: Array of points where func is stable
    """

    output = []
    last_result = np.append(x0, par0[vary_par]) # x0.copy()
    old_result = last_result[:]
    while last_result[-1] < par_max:
        old_old_result = old_result[:]
        old_result = last_result[:]

        # Secant
        u_delta = old_result - old_old_result

        # Prediction
        prediction = last_result + u_delta * delta_multiplier

        # NOTE: 0.1 shouldnt be needed
        g = lambda U: [
            *discretisation(func)(U[:-1], p={vary_par: U[-1]}),
            0.1-np.dot(prediction - U, last_result - U),  # delta v dot (v - v_tilde)
        ]

        last_result = solver(g, prediction)

        print(f"Done {last_result=} {func(last_result[:-1], p={vary_par: last_result[-1]})=}")
        output.append((last_result[-1], *last_result[:-1]))
    
    par0[vary_par] = par_max
    last_result = fsolve(discretisation(func), last_result[:-1], args=par0)
    output.append((par0[vary_par], *last_result))
    print(f"Done {last_result=} {func(last_result, p=par0)=}")

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
        delta_multiplier=0.1, # max step size
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
