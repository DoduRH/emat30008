# Import libraries
import time
import numpy as np

def perf_measure(func, num_seconds, *args):
    """Measure the performance of a function

    Args:
        func (function): Function to measure the performance of.  Called as ``func(*args)``
        num_seconds (float): Number of seconds to measure the performance over

    Returns:
        tuple: (total time, number of iterations, result of 1 call)
    """

    # Check number of seconds is valid
    # Check it can be a float
    if not np.can_cast(num_seconds, float):
        raise TypeError("num_seconds must be castable to float")
    else:
        # Convert num_seconds to float
        num_seconds = float(num_seconds)

    # Check it is greater than 0
    if num_seconds <= 0:
        raise ValueError("num_seconds must be larger than 0")

    # Initialise variables
    total_iterations = 0
    start_time = time.time()

    # Initialise mean and variance
    mean = 0
    m2 = 0

    # Run the function loop
    while time.time() - start_time < num_seconds:
        run_start = time.time()
        res = func(*args)
        total_iterations += 1

        # Update variance calculation
        last_run_time = time.time() - run_start
        d1 = last_run_time - mean
        mean += d1 / total_iterations
        d2 = last_run_time - mean
        m2 += d1 * d2

    end_time = time.time()

    total_time = end_time - start_time

    # Calculate variance from m2
    variance = m2/total_iterations

    return total_time, total_iterations, variance, res

if __name__ == "__main__":
    # Test on sleep function

    # Test with no args
    for i in [1, 4]:
        print(perf_measure(lambda: time.sleep(i), 10))
    
    # Test with args
    for i in [1, 4]:
        print(perf_measure(time.sleep, 10, i))

