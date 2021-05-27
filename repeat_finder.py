from ode_solver import solve_ode
import numpy as np

class TimePeriodNotFoundError(Exception):
    pass

def find_repeats(arr, closeArgs=None):
    """Gets estimate for repeated values in arr

    Args:
        arr (ndarray): Array of data to search through
        closeArgs (dict, optinal): Arguments to pass to np.allclose to determine if 2 points are close enough.  Defaults to None which expands to rtol=0.05, atol=0.01

    Returns:
        tuple: tuple of (0, 0, ..., -1) if search fails or (x, y, ..., period)
    """

    # arr must be floats
    if not np.can_cast(arr, float):
        raise TypeError("arr must be castable to floats")

    # closeArgs parameter
    if closeArgs is None:
        closeArgs = dict(rtol=0.05, atol=0.01)
    elif type(closeArgs) != dict:
        raise TypeError("closeArgs must be a dictionary with keys 'rtol' and/or 'atol'")

    # Reshape arr to be n by x array of numbers
    arr = np.reshape(arr, (len(arr), -1)).astype(float)

    # Round to 3DP
    rounded = np.around(arr[:,0], 3)

    # Get unique values
    unique = np.array(np.unique(rounded, axis=0, return_counts=True, return_index=True)).T

    # Get index of unique values
    # Ignore y vals
    _, index, count = unique[unique[:,2].argsort()][-1]

    output = arr[int(index)]
    
    # If half the points are the same, it is an equilibrium (period of 0)
    if count > len(arr)*0.5:
        return (*output, 0)
    
    # Has the current point gone outside of allclose range
    within_range = False
    occurences = []
    for i, row in enumerate(arr):
        if np.allclose(row, output, **closeArgs):
            # If it has been outside 'close' range
            if not within_range:
                occurences.append(i)
                within_range = True
        else:
            within_range = False

    # Must be at least 3 occurences to be considered periodic
    if len(occurences) < 3:
        return (*(0,)*len(output), -1)

    occ_array = np.array(occurences)

    # Difference in occurences
    dif = occ_array[1:] - occ_array[:-1]

    # Average of differences
    period = np.mean(dif)

    return (*output, period)


def find_period(func, t0=1, tstep=10, tmax=np.inf, *args):
    """Find estimate for period and initial conditions of func

    Args:
        func (function): Function to find period of.  Must take a single argument of t as an array. `func(t, *args)`
        t0 (int, optional): Starting value for T, must be positive. Defaults to 1.
        tstep (float, optional): Steps of t0 to take. Defaults to 1.
        tmax (float, optional): Max value for T. Defaults to np.inf.
        args: Passed to func

    Returns:
        tuple: tuple of (0, 0, ..., -1) if search fails or (x, y, ..., period)
    """
    # Clean inputs prior to use
    # t0
    if not np.can_cast(t0, float):
        raise TypeError("t0 must be castable to a float")
    else:
        t0 = float(t0)

    # tstep
    if not np.can_cast(tstep, float):
        raise TypeError("tstep must be castable to a float")
    else:
        tstep = float(tstep)

    # tmax
    if not np.can_cast(tmax, float):
        raise TypeError("tmax must be castable to a float")
    else:
        tmax = float(tmax)

    t = np.arange(0, t0, 0.1)

    *initials, period = find_repeats(func(t, *args))

    while period == -1 and t0 < tmax:
        t0 += tstep
        t = np.arange(0, t0, 0.1)
        *initials, period = find_repeats(func(t, *args))

    # If no period was found, raise error
    if period == -1:
        raise TimePeriodNotFoundError

    # Convert period from index to seconds
    period = period * 0.1
    return (*initials, period)


if __name__ == "__main__":
    # Define Lokta-Volterra equation
    alpha = 1
    delta = 0.1
    beta = 0.2

    predator_prey = lambda t, U: [
        U[0] * (1 - U[0]) - (alpha * U[0] * U[1]) / (delta + U[0]), # dx/dt
        beta * U[1] * (1 - (U[1]/U[0])), # dy/dt
    ]

    # Find ICs and period and unpack
    *initial_conditions, period = find_period(lambda t: solve_ode(predator_prey, [0.25, 0.25], t, 0.1, "rk4"))

    print(f"{initial_conditions=} {period=}")

    # Plot results to show a single orbit
    import matplotlib.pyplot as plt

    t = np.linspace(0, period, 100)
    vals = solve_ode(predator_prey, initial_conditions, t, 0.1, "RK4")

    t = np.linspace(0, 120, 1000)
    plt.plot(*solve_ode(predator_prey, [0.25, 0.25], t, 0.1, method="rk4").T, label="Periodic motion")

    plt.plot(*vals.T, label="1 Period")
    plt.scatter(initial_conditions[0], initial_conditions[1], label="Periodic Initial Conditions", color="red")
    plt.title("Plot showing a the repeat finder using the predator-prey equations")

    plt.scatter(0.25, 0.25, label="Initial Conditions")

    plt.legend(loc="upper left")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    