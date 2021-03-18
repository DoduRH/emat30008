import numpy as np

def euler_step(funcs, initial_values, t0, h):
	"""Performs 1 Euler step from t0 to t1 (t + h)

	Args:
		funcs (list): List of ode functions
		initial_values (list): List of initial values
		t0 (float): Starting value for Euler step
		h (float): Step size between t0 and t1

	Returns:
		list: List of values at t1
	"""

	# Do step for each function
	x1 = np.array([])
	for x0, func in zip(initial_values, funcs):
		np.append(x1, x0 + func(t0, *initial_values)*h)

	return x1

def RK4_step(funcs, initial_values, t0, h):
	"""Performs 1 fourth Order Runge-Kutta step from t0 to t1 (t + h)

	Args:
		funcs (list): List of ode functions
		initial_values (list): List of initial values
		t0 (float): Starting value for RK4 step
		h (float): Step size between t0 and t1

	Returns:
		list: List of values at t1
	"""
	
	# Calculate k 1-4 for each function
	k1 = np.array([])
	for func in funcs:
		k1 = np.append(k1, h * func(t0, *initial_values))

	k2 = np.array([])
	for func, k in zip(funcs, k1):
		k2 = np.append(k2, h * func(t0 + h/2, *(initial_values + k/2)))
	
	k3 = np.array([])
	for func, k in zip(funcs, k2):
		k3 = np.append(k3, h * func(t0 + h/2, *(initial_values + k/2)))
	
	k4 = np.array([])
	for func, k in zip(funcs, k3):
		k4 = np.append(k4, h * func(t0 + h, *(initial_values + k)))

	# Calulate next x's
	x1 = np.array([])
	for x0, m1, m2, m3, m4 in zip(initial_values, k1, k2, k3, k4):
		x1 = np.append(x1, x0 + (m1 + 2*m2 + 2*m3 + m4)/6)

	return x1

def solve_to(step_func, f, x0, t0, t1, hmax):
	"""Perform multiple steps between t0 and t1 with max stepsize of hmax

	Args:
		step_func (function): Function to use for each step
		f (list): List of functions to integrate over
		x0 (list): Initial conditions
		t0 (float): Starting point for integration
		t1 (float): End point for integration
		hmax (float): Max jump for each step

	Returns:
		list: List of values at t1
	"""
	t = t0
	x = x0
	while t + hmax < t1:
		x = step_func(f, x, t, hmax)
		t += hmax

	# Do final step to t1
	x = step_func(f, x, t, t1 - t)

	return x

def solve_ode(funcs, x0, t, hmax, method="euler"):
	"""Solve system of ODEs returning value at each point in t

	Args:
		funcs (list): List of functions to make the system of ODEs
		x0 (Initial conditions): Initial conditions
		t (list): Values of t to solve for
		hmax (float): Max stepsize
		method (str, optional): Method to use ("Euler" or "Rk4"). Defaults to "euler".

	Returns:
		list: List values for each t
	"""

	# get step function from method string
	method = method.lower()
	if method in methods.keys():
		# Get step function
		step_func = methods[method]
	else:
		# If string is not a recognised method
		raise ValueError(f"method must be one of {list(methods.keys())}")

	# Set up intitial variables
	x_out = np.empty((len(t), 2))
	t_start = t[0]
	# Make sure x is a list of initial conditions
	if type(x0) != np.ndarray:
		if type(x0) == list:
			x = np.array(x0)
		else:
			x = np.array([x0])
	else:
	# copy initial values list
		x = x0.copy()
	
	if type(funcs) != list:
		funcs = [funcs]

	# Solve between values t_end and t_start
	for row, t_end in enumerate(t):
		x = solve_to(step_func, funcs, x, t_start, t_end, hmax)
		x_out[row, :] = x
		# Update start range
		t_start = t_end

	return x_out

# Dictionary of available integration methods
methods = {
	"euler": euler_step,
	"rk4": RK4_step,
}