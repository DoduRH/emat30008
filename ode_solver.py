from typing import Callable


def euler_step(func: Callable[[float, float], float], x0: float, t0:float, h: float) -> float:
	'''
		Performs 1 Euler step from t0 to t1 (t + h)
	'''

	dydx = func(t0, x0)
	x1 = x0 + dydx * h

	return x1
