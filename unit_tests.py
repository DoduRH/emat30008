from repeat_finder import TimePeriodNotFoundError, find_period
from numpy.core.fromnumeric import repeat
from root_finder import find_root
import unittest
import numpy as np
from ode_solver import solve_ode
from numerical_shooting import shoot

alpha = 1
delta = 0.1
beta = 0.2

Lokta_Volterra = [
    lambda t, x, y: x * (1 - x) - (alpha * x * y) / (delta + x), # dx/dt
    lambda t, x, y: beta * y * (1 - (y/x)), # dy/dt
]

class ODETests(unittest.TestCase):
    def test_1d(self):
        """Test single ODE against analytic solution
        """
        eq = lambda t, x: x # dx/dt
        t = np.linspace(0, 10, 10)
        analytic_sol = np.exp(t)
        euler_sol = solve_ode(eq, 1, t, 0.01, "euler")[:,0]
        self.assertTrue(np.allclose(analytic_sol, euler_sol, rtol=0.1))
        rk4_sol = solve_ode(eq, 1, t, 0.1, "rk4")[:,0]
        self.assertTrue(np.allclose(analytic_sol, rk4_sol))
        pass

    def test_2d(self):
        """Test system of 2 ODEs agains analytic solution
        """
        eq = [
            lambda t, x, y: y, # x
            lambda t, x, y: -x, # y
        ]

        initial = [2, 1]

        t = np.linspace(0, 10, 10)

        # Analytic solution transposed to match solve_ode output dimensions
        analytic_sol = np.array([
            np.sin(t) - 2* (-np.cos(t)),
            np.cos(t) - 2* (np.sin(t)),
        ]).T
        # Check Euler solution is within 10% of analytic solution
        euler_sol = solve_ode(eq, initial, t, 0.01, "euler")
        self.assertTrue(np.allclose(analytic_sol, euler_sol, rtol=0.1), "Euler 2d failed")
        # Check RK4 solution is within 10% of analytic solution
        rk4_sol = solve_ode(eq, initial, t, 0.01, "rk4")
        self.assertTrue(np.allclose(analytic_sol, rk4_sol, rtol=0.1), "RK4 2d failed")
        pass

    def test_3d(self):
        """Test system of 3 ODEs agains analytic solution
        """
        pass

    def test_fewer_initials(self):
        """Check ValueError is raised when too few initial conditions are passed
        """
        eq = [
            lambda t, x, y: y, # x
            lambda t, x, y: -x, # y
        ]

        initial = 2

        t = np.linspace(0, 10, 10)

        self.assertRaises(ValueError, solve_ode, eq, initial, t, 0.1, "rk4")
        pass
    
    def test_fewer_equations(self):
        """Check ValueError is raised when too few equations are passed
        """
        eq = [
            lambda t, x, y: y, # x
        ]

        initial = [2, 1]

        t = np.linspace(0, 10, 10)

        self.assertRaises(ValueError, solve_ode, eq, initial, t, 0.1, "rk4")
        pass

    def test_undefined_arithmetic(self):
        """Check ArithmeticError is raised when equations have arithmetic error (e.g. divide-by-zero)
        """
        eq = [
            lambda t, x, y: y, # dx/dt
            lambda t, x, y: -x/ (2*x-y), # dy/dt
        ]

        initial = [1, 2]

        t = np.linspace(0, 10, 10)

        self.assertRaises(ArithmeticError, solve_ode, eq, initial, t, 0.1, "rk4")
        pass


class repeatFinderTests(unittest.TestCase):    
    def test_lokta_volterra(self):
        """Test period finding for Lokta Volterra equations
        """
        alpha = 1
        delta = 0.1
        beta = 0.2

        Lokta_Volterra = [
            lambda t, x, y: x * (1 - x) - (alpha * x * y) / (delta + x), # dx/dt
            lambda t, x, y: beta * y * (1 - (y/x)), # dy/dt
        ]

        *initials, period = find_period(lambda t: solve_ode(Lokta_Volterra, [0.25, 0.25], t, 0.1, "rk4"))

        self.assertTrue(initials, solve_ode(Lokta_Volterra, initials, [0, period], 0.1, "rk4"))
        pass


class rootFindingTests(unittest.TestCase):
    def test_linear(self):
        """Check root finding for linear equation
        """
        y = lambda x: x + 4
        root = find_root(y, 1023)
        self.assertTrue(np.allclose(y(root), 0))
        pass

    def test_quadratic(self):
        y = lambda x: x ** 2 - 4 * x + 2
        root = find_root(y, 1023)
        self.assertTrue(np.allclose(y(root), 0))
        pass
        
    def test_undefined_arithmetic(self):
        """Check ArithmeticError is raised when equations have arithmetic error (e.g. divide-by-zero)
        """
        y = lambda x: 1/x # dy/dt

        # XXX: Should this raise a more specific error than LinAlgError
        self.assertRaises(np.linalg.LinAlgError, find_root, y, 0)
        pass

class numericalShootingTests(unittest.TestCase):
    def test_lokta_volterra(self):
        """Check valid solution is found for lokta volterra equation
        """
        x, y, period = shoot(Lokta_Volterra, [0.25, 0.25])
        # XXX: Add a check for a single period?
        self.assertTrue(np.allclose([x, y], solve_ode(Lokta_Volterra, [x, y], [0, period], hmax=0.1, method="rk4")[-1]))
        pass

    def test_low_tmax(self):
        """Check suitable error is raised when no period is found
        """
        self.assertRaises(TimePeriodNotFoundError, shoot, Lokta_Volterra, [0.25, 0.25], 20)
        pass
    
    def test_incorrect_dimensions(self):
        """Check ValueError is raised when the function and initial conditions have different dimensions
        """
        pass

if __name__ == "__main__":
    unittest.main()