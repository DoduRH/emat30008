# %%
# Notebook to compare the performance of find_root and fsolve
# Import libraries
from scipy.optimize import fsolve
from root_finder import find_root
from measure_performance import perf_measure
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Create Chebyshev polynomial
cheby = np.polynomial.Chebyshev([0.1, 0.4, -0.5, 0.6])
root2 = lambda x: x ** 2 - 2

# Create tests
measurements = [
    dict(method=find_root, x0=10, func=cheby, function_name="chebychev"),
    dict(method=find_root, x0=1, func=cheby, function_name="chebychev"),
    dict(method=fsolve, x0=10, func=cheby, function_name="chebychev"),
    dict(method=fsolve, x0=1, func=cheby, function_name="chebychev"),
    dict(method=find_root, x0=1, func=root2, function_name="root2"),
    dict(method=fsolve, x0=1, func=root2, function_name="root2"),
]

# %%
# Plot function
x = np.linspace(-10, 10, 100)
plt.plot(x, cheby(x))
plt.ylim(-1, 1)
plt.show()

# %%
# Run and measure execution time for each method
results = pd.DataFrame(columns=["Root Finding Method", "Function", "Error", "Total Iterations", "Iterations/second"])

for test in measurements:
    # Get values from dictionary
    method = test['method']
    x0 = test['x0']
    func = test['func']
    func_name = test['function_name']
    # Time it
    # Run for 5 seconds to reduce variance
    total_time, total_iterations, variance, res = perf_measure(method, 1, func, x0)

    it_per_second = round(total_iterations/total_time, 3)

    results = results.append({
        "Root Finding Method": method.__name__,
        "Function": func_name,
        "Error": abs(*func(res)), # Error stored as string to lim
        "Total Iterations": total_iterations,
        "Iterations/second": it_per_second,
    }, ignore_index=True)

    # Print error and execution time
    print(f"Function {method.__name__} with function {func_name} and initial approximation {x0} with the solution {res} an error of {func(res)=}")
    print(f"ran {total_iterations} itterations in {round(total_time, 3)} seconds with a variance of {variance} giving {it_per_second} iterations/second\n")

# %%
output = results.copy()
output.Error = results.Error.apply(lambda x: "%.3g" % x)
output.to_latex("data/rootMeasurement.tex", index=False)
# %%
