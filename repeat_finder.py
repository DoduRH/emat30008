from ode_solver import solve_ode
import numpy as np

class TimePeriodNotFoundError(Exception):
    pass

def find_repeats(arr):
    """Gets estimate for repeated values in arr

    Args:
        arr (ndarray): Array of data to search through

    Returns:
        tuple: tuple of (0, 0, ..., -1) if search fails or (x, y, ..., period)
    """
    # Reshape arr to be n by x array of numbers
    arr = np.reshape(arr, (len(arr), -1))

    rounded = np.around(arr[:,0], 3)

    unique = np.array(np.unique(rounded, axis=0, return_counts=True, return_index=True)).T

    # Ignore y vals and count
    _, index, _ = unique[unique[:,2].argsort()][-1]

    output = arr[int(index)]

    occurences = []
    last = np.nan
    for i, row in enumerate(arr):
        # BUG: Potentially fails if the bottom of one wave is missed (e.g. 1, 2, and 4 are close)
        if np.allclose(row[0], output[0], 1e-2):
            # If last added was not within the last 5 itterations append it
            if last < i - 5:
                occurences.append(i)
            last = i

    occ_array = np.array(occurences)

    if len(occurences) < 3:
        return (*(0,)*len(output), -1)

    dif = occ_array[1:] - occ_array[:-1]

    period = np.mean(dif)

    return (*output, period)


def find_period(func, x0, t0=1, hmax=0.1, tstep=1, tmax=np.inf, ode_method="rk4"):
    """Find estimate for period and initial conditions of func

    Args:
        func (function): Function to find period of
        x0 (list): Initial conditions
        t0 (int, optional): Starting value for T, must be positive. Defaults to 1.
        hmax (float, optional): Hmax when integrating. Defaults to 0.1.
        tstep (float, optional): Steps of t0 to take. Defaults to 1.
        tmax (float, optional): Max value for T. Defaults to np.inf.
        ode_method (str, optional): ODE solving method. Defaults to "rk4".

    Returns:
        tuple: tuple of (0, 0, ..., -1) if search fails or (x, y, ..., period)
    """
    period = -1
    initials = (0,) * len(x0)
    while period == -1 and t0 < tmax:
        t0 += tstep
        t = np.arange(0, t0, 0.1)
        solution = solve_ode(func, x0, t, hmax, ode_method)
        *initials, period = find_repeats(solution)

    return (*initials, period)


if __name__ == "__main__":
    alpha = 1
    delta = 0.1
    beta = 0.2

    Lokta_Volterra = [
        lambda t, x, y: x * (1 - x) - (alpha * x * y) / (delta + x), # dx/dt
        lambda t, x, y: beta * y * (1 - (y/x)), # dy/dt
    ]

    print(find_period(Lokta_Volterra, [0.25, 0.25]))