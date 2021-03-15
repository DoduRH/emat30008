import numpy as np

def find_repeats(arr, decimals=3):
    rounded = np.around(arr, decimals)
    (unique, counts) = np.unique(rounded, return_counts=True)
    frequencies = np.asarray((unique, counts)).T.tolist()
    frequencies.sort(key=lambda x: x[1])
    max_val = frequencies[-1]
    return max_val[0]

if __name__ == "__main__":
    t = np.linspace(0, 20, 100)
    a = np.sin(t)
    b = np.cos(t)
    arr = np.array([a,b]).transpose()

    print(find_repeats(arr))