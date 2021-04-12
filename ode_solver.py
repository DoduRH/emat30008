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
	x1 = np.empty(len(funcs))
	for i, (x0, func) in enumerate(zip(initial_values, funcs)):
		x1[i] = x0 + func(t0, *initial_values)*h

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
	k1 = np.empty(len(funcs))
	for i, func in enumerate(funcs):
		k1[i] = h * func(t0, *initial_values)

	k2 = np.empty(len(funcs))
	for i, (func, k) in enumerate(zip(funcs, k1)):
		k2[i] = h * func(t0 + h/2, *(initial_values + k/2))
	
	k3 = np.empty(len(funcs))
	for i, (func, k) in enumerate(zip(funcs, k2)):
		k3[i] = h * func(t0 + h/2, *(initial_values + k/2))
	
	k4 = np.empty(len(funcs))
	for i, (func, k) in enumerate(zip(funcs, k3)):
		k4[i] = h * func(t0 + h, *(initial_values + k))

	# Calulate next x's
	x1 = np.empty(len(funcs))
	for i, (x0, m1, m2, m3, m4) in enumerate(zip(initial_values, k1, k2, k3, k4)):
		x1[i] = x0 + (m1 + 2*m2 + 2*m3 + m4)/6

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
	with np.errstate(divide='ignore', invalid='ignore'):
		while t + hmax < t1:
			x_old = x
			x = step_func(f, x, t, hmax)
			if not np.isfinite(x).all():
				raise ArithmeticError(f"Error with arguments {x_old} cannot continue ODE")
			t += hmax

	# Do final step to t1
	x = step_func(f, x, t, t1 - t)

	if not np.isfinite(x).all():
		raise ArithmeticError(f"Error with arguments {x_old} cannot continue ODE")

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
	x_out = np.empty((len(t), len(x)))
	
	if type(funcs) != list:
		funcs = [funcs]

	if len(funcs) != len(x):
		if len(funcs) > len(x):
			raise ValueError(f"Too many ODEs ({len(funcs)}) for initial conditions ({len(x)})")	
		else:
			raise ValueError(f"Too few ODEs ({len(funcs)}) for initial conditions ({len(x)})")

	x_out[0, :] = x
	# Solve between values t_end and t_start
	for row, t_end in enumerate(t[1:], start=1):
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