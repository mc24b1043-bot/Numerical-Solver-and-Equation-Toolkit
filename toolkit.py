"""
Numerical Solver and Equation Toolkit
Core library: root finding, linear systems, ODE solvers.
Only NumPy is used (no SciPy), so every algorithm is written from scratch.
"""
import math
import numpy as np


# ---------------------------------------------------------------- 1. ROOT FINDING
def bisection(f, a, b, tol=1e-6, max_iter=100):
    """Bisection: needs f(a)*f(b) < 0. Returns (root, iterations, history)."""
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    history = []
    for k in range(1, max_iter + 1):
        c = (a + b) / 2
        history.append(c)
        if abs(b - a) / 2 < tol or f(c) == 0:
            return c, k, history
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, max_iter, history


def newton(f, df, x0, tol=1e-10, max_iter=50):
    """Newton-Raphson: x_{k+1} = x_k - f/f'. Returns (root, iterations, history)."""
    x = x0
    history = [x]
    for k in range(1, max_iter + 1):
        d = df(x)
        if d == 0:
            raise ZeroDivisionError("f'(x) = 0, Newton-Raphson fails")
        x_new = x - f(x) / d
        history.append(x_new)
        if abs(x_new - x) < tol:
            return x_new, k, history
        x = x_new
    return x, max_iter, history


# ---------------------------------------------------------------- 2. LINEAR SYSTEMS
def gauss_solve(A, b):
    """Gaussian elimination with partial pivoting + back substitution."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    for k in range(n - 1):
        p = k + np.argmax(np.abs(A[k:, k]))          # partial pivoting
        if A[p, k] == 0:
            raise ValueError("Matrix is singular")
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        for i in range(k + 1, n):
            l = A[i, k] / A[k, k]                    # multiplier l_ik
            A[i, k:] -= l * A[k, k:]
            b[i] -= l * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):                   # back substitution
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def lu_decompose(A):
    """LU with partial pivoting: returns P, L, U such that P @ A = L @ U."""
    A = np.array(A, dtype=float)
    n = len(A)
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()
    for k in range(n - 1):
        p = k + np.argmax(np.abs(U[k:, k]))
        if U[p, k] == 0:
            raise ValueError("Matrix is singular")
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    np.fill_diagonal(L, 1.0)
    return P, L, U


def lu_solve(P, L, U, b):
    """Solve L y = P b (forward), then U x = y (backward)."""
    n = len(b)
    pb = P @ np.array(b, dtype=float)
    y = np.zeros(n)
    for i in range(n):
        y[i] = pb[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


# ---------------------------------------------------------------- 3. ODE SOLVERS
def euler(f, t0, y0, h, t_end):
    """Euler: y_{n+1} = y_n + h f(t_n, y_n). Returns arrays (t, y)."""
    n = int(round((t_end - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y


def rk4(f, t0, y0, h, t_end):
    """Classical fourth-order Runge-Kutta. Returns arrays (t, y)."""
    n = int(round((t_end - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


# ---------------------------------------------------------------- 4. RC CIRCUIT MODEL
def rc_model(R=1e3, C=1e-3, Vin=5.0):
    """RC dVc/dt + Vc = Vin  ->  returns (f, analytic, tau)."""
    tau = R * C
    f = lambda t, v: (Vin - v) / tau
    analytic = lambda t: Vin * (1 - np.exp(-np.asarray(t) / tau))
    return f, analytic, tau


# ---------------------------------------------------------------- 5. WORKFLOW (Fig. 4)
def solve(problem_type, **kw):
    """
    Dispatcher that follows the workflow of Fig. 4:
    input -> identify type -> select method -> iterate -> error check -> result.
    problem_type: 'root' | 'linear' | 'ode'
    """
    if problem_type == "root":
        f, df = kw["f"], kw.get("df")
        method = "newton" if df is not None and "x0" in kw else "bisection"
        if method == "newton":
            r, k, hist = newton(f, df, kw["x0"], kw.get("tol", 1e-10))
        else:
            r, k, hist = bisection(f, kw["a"], kw["b"], kw.get("tol", 1e-6))
        return {"method": method, "root": r, "iterations": k,
                "residual": abs(f(r)), "history": hist}

    if problem_type == "linear":
        A, b = np.array(kw["A"], float), np.array(kw["b"], float)
        P, L, U = lu_decompose(A)
        x = lu_solve(P, L, U, b)
        return {"method": "LU", "x": x, "residual": np.linalg.norm(A @ x - b)}

    if problem_type == "ode":
        step = rk4 if kw.get("method", "rk4") == "rk4" else euler
        t, y = step(kw["f"], kw["t0"], kw["y0"], kw["h"], kw["t_end"])
        out = {"method": kw.get("method", "rk4"), "t": t, "y": y}
        if "exact" in kw:
            out["max_error"] = float(np.max(np.abs(y - kw["exact"](t))))
        return out

    raise ValueError("Unknown problem type")
