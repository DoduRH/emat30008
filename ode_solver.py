from numba.core.registry import CPUDispatcher
from numba import njit

@njit
def euler_step(func, x0, t0, h):
	'''
		Performs 1 Euler step from t0 to t1 (t + h)
	'''

	dydx = func(t0, x0)
	x1 = x0 + dydx * h

	return x1

@njit
def RK4_step(func, x0, t0, h):
	'''
		Performs 1 fourth Order Runge-Kutta step from t0 to t1 (t + h)
	'''

	# Calculate k 1-4
	k1 = h * func(t0, x0)
	k2 = h * func(t0 + h/2, x0 + k1/2)
	k3 = h * func(t0 + h/2, x0 + k2/2) 
	k4 = h * func(t0 + h, x0 + k3)

	x1 = x0 + (k1 + 2*k2 + 2*k3 + k4)/6

	return x1

@njit
def solve_to(step_func, f, x0, t0, t1, hmax):
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

	if type(f) != CPUDispatcher:
		f = njit(f)

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
	"rk4": RK4_step,
}