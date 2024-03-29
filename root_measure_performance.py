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
cheby1 = lambda x: cheby(x) + 1
root2 = lambda x: x ** 2 - 2

# Create tests
measurements = [
    dict(method=find_root, x0=10, func=cheby, function_name="chebychev"),
    dict(method=find_root, x0=1, func=cheby, function_name="chebychev"),
    dict(method=fsolve, x0=10, func=cheby, function_name="chebychev"),
    dict(method=fsolve, x0=1, func=cheby, function_name="chebychev"),
    dict(method=find_root, x0=10, func=cheby1, function_name="chebychev + 1"),
    dict(method=fsolve, x0=10, func=cheby1, function_name="chebychev + 1"),
    dict(method=find_root, x0=1, func=root2, function_name="root2"),
    dict(method=fsolve, x0=1, func=root2, function_name="root2"),
]

# %%
# Plot functions
x = np.linspace(-2, 2, 100)
plt.plot(x, cheby(x), label="Chebyshev")
plt.plot(x, cheby1(x), label="Chebyshev + 1")
fsol = fsolve(cheby1, 10)
plt.scatter(fsol, cheby1(fsol), label="Fsolve Solution", color="orange")
plt.plot(x, root2(x), label="$y=x^2 - 2$")
plt.legend(loc="lower right")
plt.ylim(-2, 2)
plt.title(f"Graph of test functions")
plt.grid()
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
    # Run for 1 second to reduce variance
    total_time, total_iterations, variance, res = perf_measure(method, 1, func, x0)

    it_per_second = round(total_iterations/total_time, 3)

    # Append result to the results dataframe
    results = results.append({
        "Root Finding Method": method.__name__,
        "Function": func_name,
        "Error": abs(*func(res)),
        "Total Iterations": total_iterations,
        "Iterations/second": it_per_second,
    }, ignore_index=True)

    # Print error and execution time
    print(f"Function {method.__name__} with function {func_name} and initial approximation {x0} with the solution {res} an error of {func(res)=}")
    print(f"ran {total_iterations} itterations in {round(total_time, 3)} seconds with a variance of {variance} giving {it_per_second} iterations/second\n")

# %%
# Save results
output = results.copy()
output.Error = results.Error.apply(lambda x: "%.3g" % x) # Convert errors to strings to limit decimals
output.to_latex("data/rootMeasurement.tex", index=False) # Save output directly to a latex file
# %%
