# Emat30008 - Scientific Computing

This software package to performs numerical methods in Python for solving ODEs, root finding, numerical shooting, repeat finder, numerical continuation and the heat equation PDE as well as a tool for measuring the performance of a function.  

## What's Included
### ODE
Euler's method and Runge-Kutta step methods are implemented for solving ODEs.  These methods are compared for accuracy and run time before discussing the strengths and weaknesses of their implementation.  

### Shooting
Numerical shooting as well as a repeat finding method for finding initial conditions for the numerical shooting are implemented.  

### Numerical Continuation
Numerical continuation with pseudo-arclength continuation is implemented and used to find the roots of $y=x^3 - x + c$ for $-2 \leq c \leq 2$.

### PDE
PDE solving for the PDE $\frac{\partial u}{\partial t} = \kappa \frac{\partial ^2u}{\partial t^2}$ using forward Euler, backward Euler and Crank-Nicholson step methods are implemented. 

## Installation

See ```requirements.txt``` for the required modules.  Python 3.8 or greater is required.  The code has been fully tested with Python 3.8.8.

## Usage

The following files contain the main functionality for the software

- ODE Solver - ```ode_solver.py```
- Shooting - ```numerical_shooting.py```
- Continuation - ```numerical_continuation.py```
- PDE - ```pde_solver.py```
- Unit tests - ```unit_tests.py```

The other scripts use the files listed above to create visualisations, measure performance and measure accuracy.