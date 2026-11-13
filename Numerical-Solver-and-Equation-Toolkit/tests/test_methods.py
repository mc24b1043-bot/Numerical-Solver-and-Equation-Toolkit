import numpy as np
from modules.numerical_analysis import bisection,gauss_elimination,integrate
from modules.differential_equations import solve_ode
from modules.signal_processing import combine_signals,convolve_signals,spectrum
def test_bisection(): assert abs(bisection("x**3-x-2",1,2)[0]-1.5213797)<1e-6
def test_gauss(): assert np.allclose(gauss_elimination([[2,1],[1,3]],[5,6]),[1.8,1.4])
def test_integrate(): assert abs(integrate("x**2",0,2)[0]-8/3)<1e-8
def test_rk4():
 t,y=solve_ode("-2*y",1,0,1,.1);assert abs(y[-1]-np.exp(-2))<1e-5


def test_signal_operations():
    assert np.allclose(combine_signals([1, 2], [3, 4]), [4, 6])
    assert np.allclose(combine_signals([1, 2], [3, 4], "Multiplication"), [3, 8])
    assert np.allclose(convolve_signals([1, 2], [1, 1]), [1, 3, 2])

def test_fft_frequency_peak():
    from modules.signal_processing import make_signal
    _, y = make_signal("Sine", 1, 5, 100, 2)
    f, a = spectrum(y, 100)
    assert abs(f[np.argmax(a)] - 5) < 0.6
