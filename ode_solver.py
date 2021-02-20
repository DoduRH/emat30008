from typing import Callable


def euler_step(func: Callable[[float, float], float], x0: float, t0:float, h: float) -> float:
	'''
		Performs 1 Euler step from t0 to t1 (t + h)
	'''

	dydx = func(t0, x0)
	x1 = x0 + dydx * h

	return x1

def solve_to(step_func: Callable, f: Callable[[float, float], float], x0: float, t0: float, t1: float, hmax: float) -> float:
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
