from repeat_finder import TimePeriodNotFoundError, find_period
from root_finder import find_root, jacobian_matrix
import unittest
import numpy as np
from ode_solver import solve_ode
from numerical_shooting import shoot
from scipy.optimize import fsolve
from pde_solver import solve_pde
from measure_performance import perf_measure
import time

def Lokta_Volterra(t, U, p):
    x = U[0] * (1 - U[0]) - (p['alpha'] * U[0] * U[1]) / (p['delta'] + U[0]) # dx/dt
    y = p['beta'] * U[1] * (1 - (U[1]/U[0])) # dy/dt

    return [x, y]

class performanceMeasurementTests(unittest.TestCase):
    def test_perf_measure_quick(self):
        """Test performance measurement with time.sleep
        """
        func = time.sleep
        sleep_seconds = 0.1
        run_seconds = 1
        total_time, number_of_iterations, variance, result = perf_measure(func, run_seconds, sleep_seconds)

        self.assertEqual(None, result)
        self.assertAlmostEqual(number_of_iterations, run_seconds/sleep_seconds)
        pass

    def test_perf_measure_slow(self):
        """Test performance measurement with time.sleep and a return value
        """
        def func(x):
            time.sleep(x)
            return "Return Value"

        sleep_seconds = 0.5
        run_seconds = 1
        total_time, number_of_iterations, variance, result = perf_measure(func, run_seconds, sleep_seconds)

        self.assertEqual(func(0), result)
        self.assertAlmostEqual(number_of_iterations, run_seconds/sleep_seconds)
        pass

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
        eq = lambda t, U: [
            U[1], # x
            -U[0], # y
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
        # TODO: Add test
        pass

    def test_undefined_arithmetic(self):
        """Check ArithmeticError is raised when equations have arithmetic error (e.g. divide-by-zero)
        """
        eq = lambda t, U: [
            U[1], # dx/dt
            -U[0]/ (2*U[0]-U[1]), # dy/dt
        ]

        initial = [1, 2]

        t = np.linspace(0, 10, 10)

        p = dict(
            alpha = 1,
            delta = 0.1,
            beta = 0.2,
        )

        self.assertRaises(ArithmeticError, solve_ode, eq, initial, t, 0.1, "rk4")
        pass

    def test_uncastable_x0(self):
        """Check ValueError is raised when initial conditions can't be cast to float
        """
        eq = lambda t, U: [
            U[1], # dx/dt
            -U[0]/ (2*U[0]-U[1]), # dy/dt
        ]

        initial = [1, "2"]

        t = np.linspace(0, 10, 10)

        p = dict(
            alpha = 1,
            delta = 0.1,
            beta = 0.2,
        )

        self.assertRaises(ValueError, solve_ode, eq, initial, t, 0.1, "rk4")
        pass


class repeatFinderTests(unittest.TestCase):    
    def test_lokta_volterra(self):
        """Test period finding for Lokta Volterra equations
        """
        p = dict(
            alpha = 1,
            delta = 0.1,
            beta = 0.2,
        )

        *initials, period = find_period(lambda t: solve_ode(Lokta_Volterra, [0.25, 0.25], t, 0.1, "rk4", [p]))

        self.assertTrue(np.allclose(initials, solve_ode(Lokta_Volterra, initials, [0, period], 0.1, "rk4", [p])[-1], rtol=0.05))
        pass

    def test_low_tmax(self):
        """Test period finding for Lokta Volterra equations when low tmax is specified
        """
        params = dict(
            alpha = 1,
            delta = 0.1,
            beta = 0.2,
        )

        #*initials, period = find_period(lambda t: solve_ode(Lokta_Volterra, [0.25, 0.25], t, 0.1, "rk4"))

        self.assertRaises(TimePeriodNotFoundError, find_period, lambda t: solve_ode(Lokta_Volterra, [0.25, 0.25], t, 0.1, "rk4", [params]), tmax=50)
        pass


class jacobianFindingTests(unittest.TestCase):
    def test_linear(self):
        """Check jacobian matrix for 1d linear solve
        """
        f = lambda U: [2 * U[0]]
        exact = lambda x: 2

        self.assertTrue(np.allclose(jacobian_matrix(f, [5]), exact(5)))

        pass

    def test_quadratic(self):
        """Check jacobian matrix for 1d quadratic
        """
        f = lambda U: [2 * U[0] ** 2 + 3 * U[0] + 6]
        exact = lambda x: 4 * x + 3

        self.assertTrue(np.allclose(jacobian_matrix(f, [5]), exact(5)))
        pass

    def test_sin(self):
        """Check jacobian matrix for 1d sin wave
        """
        f = lambda U: [np.sin(U[0])]
        exact = lambda x: np.cos(x)

        self.assertTrue(np.allclose(jacobian_matrix(f, [5]), exact(5)))
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

        # NOTE: Should this raise a more specific error than ArithmeticError
        self.assertRaises(ArithmeticError, find_root, y, 0)
        pass

class numericalShootingTests(unittest.TestCase):
    def test_lokta_volterra(self):
        """Check valid solution is found for lokta volterra equation
        """
        params = dict(
            alpha = 1,
            delta = 0.1,
            beta = 0.2,
        )

        x, y, period = shoot(Lokta_Volterra, [0.25, 0.25], ODEparams=[params])
        # NOTE: Add a check for a single period?
        self.assertTrue(np.allclose([x, y], solve_ode(Lokta_Volterra, [x, y], [0, period], hmax=0.1, method="rk4", ODEparams=[params])[-1]))
        pass

    def test_low_tmax(self):
        """Check suitable error is raised when no period is found
        """
        params = dict(
            alpha = 1,
            delta = 0.1,
            beta = 0.2,
        )

        self.assertRaises(TimePeriodNotFoundError, shoot, Lokta_Volterra, [0.25, 0.25], tmax=20, ODEparams=[params])
        pass
    
    def test_hopf_bifurcation(self):
        """Check valid solution is found for Hopf bifurcation equation
        """
        
        params = dict(
            beta = 0.5,
            sigma = -1,
        )

        hopf = lambda t, U, p: [
            p['beta'] * U[0] -        U[1] + p['sigma'] * U[0] * (U[0]**2 + U[1]**2),
            U[1] + p['beta'] * U[1] + p['sigma'] * U[1] * (U[0]**2 + U[1]**2),
        ]

        x, y, period = shoot(hopf, [2, 4], ODEparams=[params], solver=fsolve)

        # Ensure solution after 1 period is within 5% of the starting value
        self.assertTrue(np.allclose([x, y], solve_ode(hopf, [x, y], [0, period], hmax=0.1, method="rk4", ODEparams=[params])[-1], rtol=0.05))

        pass

class pdeSolverTests(unittest.TestCase):
    def PDEsetup(self):
        """Return parameters for the PDE under test
        """
        # Set problem parameters/functions
        kappa = 1.0   # diffusion constant
        L=1.0         # length of spatial domain
        T=0.5         # total time to solve for

        def u_I(x):
            # initial temperature distribution
            y = np.sin(np.pi*x/L)
            return y

        def u_exact(x,t):
            # the exact solution
            y = np.exp(-kappa*(np.pi**2/L**2)*t)*np.sin(np.pi*x/L)
            return y

        # Set numerical parameters
        mx = 25     # number of gridpoints in space
        mt = 1000   # number of gridpoints in time

        analytic = u_exact(np.linspace(0, L, mx+1), T)

        return kappa, L, T, u_I, mx, mt, analytic

    def testForwardEuler(self):
        """Check forwardEuler method is close to analytic solution
        """
        # Get PDE from PDE setup
        kappa, L, T, u_I, mx, mt, analytic = self.PDEsetup()

        # run solve_pde
        numericSolution = solve_pde(mx, mt, L, T, u_I, kappa, lambda x, t: 0, "forwardEuler", find_root)

        # Compare it to analytic solution
        self.assertTrue(np.allclose(numericSolution, analytic, atol=1e-3))

        pass

    def testBackwardEuler(self):
        """Check backwardEuler method is close to analytic solution
        """

        # Get PDE from PDE setup
        kappa, L, T, u_I, mx, mt, analytic = self.PDEsetup()
        
        # run solve_pde
        numericSolution = solve_pde(mx, mt, L, T, u_I, kappa, lambda x, t: 0, "backwardEuler", find_root)

        # Compare it to analytic solution
        self.assertTrue(np.allclose(numericSolution, analytic, atol=2e-3))
        pass

    def testCrankNicholson(self):
        """Check crankNicholson method is close to analytic solution
        """
        # Get PDE from PDE setup
        kappa, L, T, u_I, mx, mt, analytic = self.PDEsetup()
        
        # run solve_pde
        numericSolution = solve_pde(mx, mt, L, T, u_I, kappa, lambda x, t: 0, "crankNicholson", find_root)

        # Compare it to analytic solution
        self.assertTrue(np.allclose(numericSolution, analytic, atol=1e-3))
        pass


if __name__ == "__main__":
    unittest.main()