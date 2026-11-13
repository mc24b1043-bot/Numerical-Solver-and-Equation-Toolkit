# Numerical Solver and Equation Toolkit

**An Integrated Platform for Numerical Computation, Linear System Analysis and Signal Processing**

A modular Streamlit scientific-computing application for B.Tech Mathematics and Computing students. It connects numerical algorithms with dynamic-system and signal analysis.

## Problem statement and objectives
Engineering models require approximate solutions, stability checks and visual interpretation across multiple subjects. This toolkit combines numerical analysis, differential equations, linear systems and signal processing.

## Academic integration
- Numerical Analysis: roots, Gaussian elimination, LU factorization, Lagrange interpolation, quadrature, finite differences and ODE methods.
- Linear Systems and Signal Processing: transfer functions, poles/zeros, responses, Bode analysis, sampling, FFT and filtering.
- Integrated RLC workflow: physical parameters produce an ODE; numerical response is computed; transfer-function poles explain stability.

## Features
- Bisection, Newton-Raphson and secant with iteration tables.
- Gaussian elimination (partial pivot), LU, interpolation, trapezoid/Simpson integration, finite differences.
- Euler, modified Euler (Heun), and RK4 scalar ODE comparison.
- Transfer function poles, zeros, stability, impulse/step and Bode plots.
- Signal generation, reproducible Gaussian noise, Butterworth filtering and FFT.
- Mass-spring-damper, series RLC and first-order simulators.
- Rule-based recommendations; downloadable ReportLab PDF summaries.

## Technology
Python 3.11+, Streamlit, NumPy, SciPy, SymPy, Plotly, Pandas, Matplotlib, ReportLab.

## Architecture
```
app.py
modules/ numerical_analysis.py, linear_systems.py, signal_processing.py,
        differential_equations.py, system_simulator.py
utils/ plotting.py, error_analysis.py, validation.py, pdf_generator.py
assets/ logo.png
tests/ test_methods.py
```

## Mathematics summary
Bisection halves a sign-changing interval. Newton uses x(n+1)=x(n)-f(x(n))/f'(x(n)); secant estimates the derivative from two points. Gaussian elimination uses partial pivoting; LU factors A=LU. Lagrange interpolation constructs a polynomial through data. Trapezoid and Simpson approximate integrals; finite differences approximate derivatives. Euler, Heun and RK4 advance ODE values with increasing order of accuracy.

Transfer-function denominator roots are poles. A continuous-time system is asymptotically stable when all pole real parts are negative. FFT decomposes sampled data into frequency bins. Butterworth filtering attenuates frequencies outside a selected passband.

## Installation and launch (Windows PowerShell)
```powershell
cd .\Numerical-Solver-and-Equation-Toolkit
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Open the localhost URL printed by Streamlit (typically http://localhost:8501). Python 3.11 or newer is recommended.

## Testing
```powershell
pytest -q
```
Tests cover the cubic root x�-x-2, a 2x2 linear system, integral x� on [0,2], and RK4 on y'=-2y.

## Screenshots
Capture the dashboard and each analysis module after launching the app, then add them here.

## Future scope
Vector ODE systems, adaptive error control, discrete-time transfer functions, CSV signal import, report plots, and configurable forcing functions.

## References
Burden, Faires & Burden, *Numerical Analysis*; Ogata, *Modern Control Engineering*; SciPy docs: https://docs.scipy.org/doc/scipy/; SymPy: https://docs.sympy.org/; Streamlit: https://docs.streamlit.io/.

## Viva presentation
For a series RLC circuit, Kirchhoff's law gives Lq''+Rq'+q/C=V(t). The Laplace transform gives Q(s)/V(s)=1/(Ls�+Rs+1/C). Denominator roots predict stability; numerical integration generates charge/current curves; signal tools aid interpretation. Recommendations are transparent rules, not machine learning.
