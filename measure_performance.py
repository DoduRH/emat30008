# Import libraries
import time

def perf_measure(func, num_seconds, *args):
    """Measure the performance of a function

    Args:
        func (function): Function to measure the performance of
        num_seconds (float): Number of seconds to measure the performance over
        args: Arguments to call func with

    Returns:
        tuple: (total time, number of iterations, result of 1 call)
    """

    if num_seconds <= 0:
        raise ValueError("num_seconds must be larger than 0")

    total_iterations = 0
    start_time = time.time()
    while time.time() - start_time < num_seconds:
        res = func(*args)
        total_iterations += 1
    end_time = time.time()

    total_time = end_time - start_time

    return total_time, total_iterations, res

if __name__ == "__main__":
    # Test on sleep function
    for i in [1, 4]:
        print(perf_measure(time.sleep, 10, i))
