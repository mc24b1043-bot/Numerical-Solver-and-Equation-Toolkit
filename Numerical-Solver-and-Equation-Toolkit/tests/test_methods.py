import numpy as np
from modules.numerical_analysis import bisection,gauss_elimination,integrate
from modules.differential_equations import solve_ode
def test_bisection(): assert abs(bisection("x**3-x-2",1,2)[0]-1.5213797)<1e-6
def test_gauss(): assert np.allclose(gauss_elimination([[2,1],[1,3]],[5,6]),[1.8,1.4])
def test_integrate(): assert abs(integrate("x**2",0,2)[0]-8/3)<1e-8
def test_rk4():
 t,y=solve_ode("-2*y",1,0,1,.1);assert abs(y[-1]-np.exp(-2))<1e-5
