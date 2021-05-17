import numpy as np

def euler_step(ODE, initial_values, t0, h, fargs=None):
	"""Performs 1 Euler step from t0 to t1 (t + h)

	Args:
		ODE (function): ODE function that returns a list.  Called as ODE(t, initial_values, *fargs)
		initial_values (list): List of initial values
		t0 (float): Starting value for Euler step
		h (float): Step size between t0 and t1
		fargs (list, optional): Arguments to pass to ODE. Defaults to no arguments.

	Returns:
		list: List of values at t1
	"""
	if fargs is None:
		fargs = []

	# Do step for each function
	x1 = initial_values + np.array(ODE(t0, initial_values, *fargs))*h

	return x1

def RK4_step(ODE, initial_values, t0, h, fargs=None):
	"""Performs 1 fourth Order Runge-Kutta step from t0 to t1 (t + h)

	Args:
		ODE (function): ODE function that returns a list.  Called as ODE(t, initial_values, *fargs)
		initial_values (list): List of initial values
		t0 (float): Starting value for RK4 step
		h (float): Step size between t0 and t1
		fargs (list, optional): Arguments to pass to ODE. Defaults to no arguments.

	Returns:
		list: List of values at t1
	"""
	if fargs is None:
		fargs = []
	
	# Calculate k 1-4 for each function
	k1 = h * np.array(ODE(t0, initial_values, *fargs))

	k2 = h * np.array(ODE(t0 + h/2, (initial_values + k1/2), *fargs))
	
	k3 = h * np.array(ODE(t0 + h/2, (initial_values + k2/2), *fargs))
	
	k4 = h * np.array(ODE(t0 + h, (initial_values + k3), *fargs))

	# Calulate next x's
	x1 = initial_values + (k1 + 2*k2 + 2*k3 + k4)/6

	return x1

def solve_to(step_func, ODE, x0, t0, t1, hmax, ODEparams=None):
	"""Perform multiple steps between t0 and t1 with max stepsize of hmax

	Args:
		step_func (function): Function to use for each step
		ODE (function): ODE function that returns a list.  Called as ODE(t, initial_values, *fargs)
		x0 (list): Initial conditions
		t0 (float): Starting point for integration
		t1 (float): End point for integration
		hmax (float): Max jump for each step
        ODEparams (list, optional): Optional extra parameters to pass to the ODE. Defaults to no arguments

	Returns:
		list: List of values at t1
	"""

	if ODEparams is None:
		ODEparams = []

	t = t0
	x = x0
	with np.errstate(divide='ignore', invalid='ignore'):
		while t + hmax < t1:
			x_old = x
			x = step_func(ODE, x, t, hmax, ODEparams)
			if not np.isfinite(x).all():
				raise ArithmeticError(f"Error with arguments {x_old} cannot continue ODE")
			t += hmax

	# Do final step to t1
	x = step_func(ODE, x, t, t1 - t, ODEparams)

	if not np.isfinite(x).all():
		raise ArithmeticError(f"Error with arguments {x_old} cannot continue ODE")

	return x

def solve_ode(funcs, x0, t, hmax, method="euler", ODEparams=None):
	"""Solve system of ODEs returning value at each point in t

	Args:
		ODE (function): ODE function that returns a list.  Called as ODE(t, initial_values, *fargs)
		x0 (Initial conditions): Initial conditions
		t (list): Values of t to solve for
		hmax (float): Max stepsize
		method (str, optional): Method to use ("Euler" or "Rk4"). Defaults to "euler".
        ODEparams (list, optional): Optional extra parameters to pass to the ODE. Defaults to no arguments

	Returns:
		list: List values for each t
	"""

	if ODEparams is None:
		ODEparams = []

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
	if np.can_cast(np.array(x0), float):
		x = np.array(x0, dtype=float).flatten()
	else:
		raise ValueError(f"x0 cannot be cast to float, recieved '{x0}'")

	x_out = np.empty((len(t), len(x)))

	x_out[0, :] = x

	# Solve between values t_end and t_start
	for row, t_end in enumerate(t[1:], start=1):
		x = solve_to(step_func, funcs, x, t_start, t_end, hmax, ODEparams)
		x_out[row, :] = x
		# Update start range
		t_start = t_end

	return x_out

# Dictionary of available integration methods
methods = {
	"euler": euler_step,
	"rk4": RK4_step,
}