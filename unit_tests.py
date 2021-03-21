import unittest
import numpy as np
from ode_solver import solve_ode
from numerical_shooting import shoot

def Lokta_Volterra(t, x, y):
    """Lokta-Volterra function

    Args:
        t (float): time
        x (float): x
        y (float): y

    Returns:
        list: x, y at n+1
    """
    # Setup equations
    # Lokta Volterra variables
    alpha = 1
    delta = 0.1
    beta = 0.2

    # Lokta-Volterra equations
    x1 =  x * (1 - x) - (alpha * x * y) / (delta + x) # dx/dt
    y1 = beta * y * (1 - (y/x)) # dy/dt

    return [x1, y1]

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

        # FIXME: This test fails
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

        # FIXME: This test fails
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

        # FIXME: This test fails
        self.assertRaises(ArithmeticError, solve_ode, eq, initial, t, 0.1, "rk4")
        pass

class numericalShootingTests(unittest.TestCase):
    def test_lokta_volterra(self):
        """Check valid solution is found for lokta volterra equation
        """
        pass

    def test_low_tmax(self):
        """Check suitable error is raised when no period is found
        """
        pass
    
    def test_incorrect_dimensions(self):
        """Check ValueError is raised when the function and initial conditions have different dimensions
        """
        pass

if __name__ == "__main__":
    unittest.main()