from typing import Callable, List


def euler_step(func, x0, t0, h):
	# type: (Callable, float, float, float) -> float
	'''
		Performs 1 Euler step from t0 to t1 (t + h)
	'''

	dydx = func(t0, x0)
	x1 = x0 + dydx * h

	return x1

def solve_to(step_func, f, x0, t0, t1, hmax):
	# type: (Callable, Callable, float, float, float, float) -> float
	'''
		Performs integration steps using step function on f from t0 to t1 in steps no larger than hmax
	'''
	t = t0
	x = x0
	while t + hmax < t1:
		x = step_func(f, x, t, hmax)
		t += hmax

	# Do final step to t1
	x = step_func(f, x, t, t1 - t)

	return x

def solve_ode(f, x0, t, hmax, method="euler"):
	# type: (Callable, float, List[float], float, str) -> List[float]
	'''
		generates a series of numerical solution estimates for f at points in t
	'''

	# get step function from method string
	method = method.lower()
	if method in methods.keys():
		# Get step function
		step_func = methods[method]
	else:
		# If string is not a recognised method
		raise ValueError(f"method must be one of {list(methods.keys())}")

	# Set up intitial variables
	x_out = []
	t_start = t[0]
	x = x0

	# Solve between values t_end and t_start
	for t_end in t:
		x = solve_to(step_func, f, x, t_start, t_end, hmax)
		x_out.append(x)
		# Update start range
		t_start = t_end

	return x_out

# Dictionary of available integration methods
methods = {
	"euler": euler_step,
}