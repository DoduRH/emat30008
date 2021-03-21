import unittest
import numpy as np
from ode_solver import solve_ode


class ODETests(unittest.TestCase):
    def test_1d(self):
        eq = lambda t, x: x # dx/dt
        t = np.linspace(0, 10, 10)
        analytic_sol = np.exp(t)
        euler_sol = solve_ode(eq, 1, t, 0.01, "euler")[:,0]
        self.assertTrue(np.allclose(analytic_sol, euler_sol, rtol=0.1))
        rk4_sol = solve_ode(eq, 1, t, 0.1, "rk4")[:,0]
        self.assertTrue(np.allclose(analytic_sol, rk4_sol))
        pass

    def test_2d(self):
        pass

    def test_3d(self):
        pass

    def test_incorrect_dimensions(self):
        pass

    def test_no_solution(self):
        pass

if __name__ == "__main__":
    unittest.main()