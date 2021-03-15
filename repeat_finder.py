import numpy as np

def find_repeats(arr, abs_tol=0.01, show_errors=False):
    rounded = arr

    # Loop through each pair in the array
    for i, a in enumerate(rounded):
        while i < rounded.shape[0] and np.allclose(rounded[i], a, atol=abs_tol):
            i += 1
        for b in rounded[i:]:
            if np.allclose(a, b, atol=abs_tol):
                return a
    
    if show_errors:
        print("No solution found")

    return np.array([0, 0])

if __name__ == "__main__":
    t = np.linspace(0, 20, 100)
    a = np.sin(t)
    b = np.cos(t)
    arr = np.array([a,b]).transpose()

    print(find_repeats(arr))