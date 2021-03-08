import numpy as np

def euler_step(funcs, initial_values, t0, h):
	'''
		Performs 1 Euler step from t0 to t1 (t + h)
	'''
	# Do step for each function
	x1 = []
	for x0, func in zip(initial_values, funcs):
		x1.append(x0 + func(t0, *initial_values)*h)

	return x1

def RK4_step(funcs, initial_values, t0, h):
	'''
		Performs 1 fourth Order Runge-Kutta step from t0 to t1 (t + h)
	'''

	initial_values = np.array(initial_values)
	
	# Calculate k 1-4 for each function
	k1 = []
	for func in funcs:
		k1.append(h * func(t0, *initial_values))

	k2 = []
	for func, k in zip(funcs, k1):
		k2.append(h * func(t0 + h/2, *(initial_values + k/2)))
	
	k3 = []
	for func, k in zip(funcs, k2):
		k3.append(h * func(t0 + h/2, *(initial_values + k/2)))
	
	k4 = []
	for func, k in zip(funcs, k3):
		k4.append(h * func(t0 + h, *(initial_values + k)))

	# Calulate next x's
	x1 = []
	for x0, m1, m2, m3, m4 in zip(initial_values, k1, k2, k3, k4):
		x1.append(x0 + (m1 + 2*m2 + 2*m3 + m4)/6)

	return x1

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

def solve_ode(funcs, x0, t, hmax, method="euler"):
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
	# Make sure x is a list of initial conditions
	if type(x0) != list:
		x0 = [x0]
	# copy initial values list
	x = x0.copy()
	
	if type(funcs) != list:
		funcs = [funcs]

	# Solve between values t_end and t_start
	for t_end in t:
		x = solve_to(step_func, funcs, x, t_start, t_end, hmax)
		x_out.append(x)
		# Update start range
		t_start = t_end

	return x_out

# Dictionary of available integration methods
methods = {
	"euler": euler_step,
	"rk4": RK4_step,
}