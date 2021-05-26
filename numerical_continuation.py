from root_finder import find_root
import numpy as np

def continuation(func, x0, par0, vary_par, par_max, discretisation=lambda x: x, solver=find_root):
    """Finds points where the function is stable

    Args:
        func (function): Function to find the line for
        x0 (List): Initial conditions
        par0 (Dict): Initial parameters
        vary_par (str): Key of dictionary to vary
        par_max (float): Max value of var_par
        discretisation (function, optional): Discretisation function. Defaults to passing func through.
        solver (function, optional): Solver to use. Defaults to find_root.

    Returns:
        array: Array of points where func is stable
    """

    # Calculate v0 and v1 to start the process
    v = []
    v.append(np.array((*solver(discretisation(func), x0, args=({vary_par: par0[vary_par]})), par0[vary_par])))
    v.append(np.array((*solver(discretisation(func), v[0][:-1], args=({vary_par: v[0][-1] + 0.1})), v[0][-1] + 0.1)))

    while v[-1][-1] < par_max:
        # Calculate the secant
        secant = v[-1] - v[-2]

        # Make a prediction 
        predict = v[-1] + secant

        # Construct g
        g = lambda Vn: [
            *discretisation(func)(Vn[:-1], {vary_par: Vn[-1]}),
            np.dot(Vn - predict, secant),
        ]

        # Solve and append to v
        v.append(solver(g, predict))

    return v

if __name__ == "__main__":
    x0 = [0.2, 0.2]

    # Hopf Bifurcation
    hopf = lambda t, U, p: [
                p['beta'] * U[0] - U[1] - U[0] * (U[0] ** 2 + U[1] ** 2),
        U[0] +  p['beta'] * U[1] -        U[1] * (U[0] ** 2 + U[1] ** 2),
    ]

    hopfPar = dict(
        beta = -1,
    )

    hopfInitial = [
        1.5,
        1.5,
    ]

    from scipy.optimize import fsolve

    cubicInitial = [1.5]
    cubicPar = dict(c=-2)
    cubic = lambda U, p: [
        U[0] ** 3 - U[0] + p['c']
    ]

    # Continuation on the cubic equation
    results = continuation(
        cubic,  # the equation to use
        cubicInitial,  # the initial state
        cubicPar,  # the initial parameters
        vary_par="c",  # the parameter to vary
        par_max=2,
        discretisation=lambda x: x,  # the discretisation to use
        solver=fsolve,  # the solver to use
    )

    # Continuation on the hopf bifurcation equation
    # results = continuation(
    #     hopf,  # the ODE to use
    #     hopfInitial,  # the initial state
    #     hopfPar,  # the initial parameters
    #     vary_par="beta",  # the parameter to vary
    #     par_max=2,
    #     delta_multiplier=0.1, # max step size
    #     discretisation=lambda f: lambda g, p: shoot(f=f, initial=g, approximate_solution=g, ODEparams=[p]),  # the discretisation to use
    #     solver=fsolve,  # the solver to use
    # )
    
    import matplotlib.pyplot as plt
    import numpy as np

    r = np.array(results).T
    x = r[0]
    ys = r[1]

    plt.xlabel("C")
    plt.ylabel("X")
    plt.title("Graph showing numerical continuation on $y=x^3 - x + c$")

    plt.plot(x, ys)
    plt.scatter(x, ys)

    plt.show()
