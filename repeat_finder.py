import numpy as np

class TimePeriodNotFoundError(Exception):
    pass

def find_repeats(arr, abs_tol=0.01, period_tol=None, show_errors=False):
    rounded = arr

    if period_tol is None:
        period_tol = abs_tol * 5
    # Loop through each pair in the array
    for i, a in enumerate(rounded):
        period_start = i
        while i < rounded.shape[0] and np.allclose(rounded[i], a, atol=abs_tol):
            i += 1
        for b in rounded[i:]:
            i += 1
            if np.allclose(a, b, atol=abs_tol):
                # Check period of oscillation works for the remaining periods
                period = i - period_start
                period_works = True
                # If only 1 period is found, raise TimePeriodNotFoundError
                if rounded[period_start::period].shape[0] <= 2:
                    raise TimePeriodNotFoundError("Time period not found")
                for x in rounded[period_start::period]:
                    if not np.allclose(x, a, atol=period_tol):
                        period_works = False
                        break
                if period_works:
                    return (a, period)
    
    if show_errors:
        print("No solution found")

    return np.array([0, 0])

if __name__ == "__main__":
    t = np.linspace(0, 20, 100)
    a = np.sin(t)
    b = np.cos(t)
    arr = np.array([a,b]).transpose()

    print(find_repeats(arr))