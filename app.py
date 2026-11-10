import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from toolkit import (
    bisection, newton, gauss_solve, lu_decompose, lu_solve,
    euler, rk4, rc_model
)

st.set_page_config(
    page_title="Numerical Methods Toolkit",
    page_icon="∑",
    layout="wide"
)

st.title("Numerical Methods Toolkit")
st.caption("An interactive solver and visualization platform")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a module",
        ["Root Finding", "Linear Systems", "ODE Solver", "RC Circuit"]
    )
    st.divider()
    st.caption("Algorithms are implemented in toolkit.py.")

def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)

if page == "Root Finding":
    st.header("Root Finding")
    st.write("Find an approximate solution of f(x) = 0.")
    equation = st.selectbox(
        "Choose an example equation",
        ["x² - 2", "x³ - x - 2", "cos(x) - x"]
    )

    if equation == "x² - 2":
        f = lambda x: x**2 - 2
        df = lambda x: 2*x
        default_a, default_b, default_x0 = 1.0, 2.0, 1.5
    elif equation == "x³ - x - 2":
        f = lambda x: x**3 - x - 2
        df = lambda x: 3*x**2 - 1
        default_a, default_b, default_x0 = 1.0, 2.0, 1.5
    else:
        f = lambda x: np.cos(x) - x
        df = lambda x: -np.sin(x) - 1
        default_a, default_b, default_x0 = 0.0, 1.0, 0.5

    col1, col2, col3 = st.columns(3)
    method = col1.selectbox("Method", ["Bisection", "Newton-Raphson"])
    tol = col2.number_input("Tolerance", min_value=1e-12, max_value=1e-2,
                            value=1e-6, format="%.1e")
    max_iter = col3.number_input("Maximum iterations", 1, 500, 100)

    if method == "Bisection":
        c1, c2 = st.columns(2)
        a = c1.number_input("Left endpoint (a)", value=default_a)
        b = c2.number_input("Right endpoint (b)", value=default_b)
    else:
        x0 = st.number_input("Initial guess (x₀)", value=default_x0)

    if st.button("Solve", type="primary"):
        try:
            if method == "Bisection":
                root, iterations, history = bisection(
                    f, a, b, tol=float(tol), max_iter=int(max_iter)
                )
            else:
                root, iterations, history = newton(
                    f, df, x0, tol=float(tol), max_iter=int(max_iter)
                )

            st.success("Calculation completed.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Approximate root", f"{root:.12g}")
            m2.metric("Iterations", iterations)
            m3.metric("Residual |f(root)|", f"{abs(f(root)):.3e}")

            history_df = pd.DataFrame({
                "Iteration": range(len(history)),
                "Approximation": history,
                "Absolute error (from final root)": [abs(v-root) for v in history]
            })
            st.subheader("Iteration table")
            st.dataframe(history_df, use_container_width=True)

            fig, ax = plt.subplots()
            ax.plot(range(len(history)), [abs(f(v)) for v in history], marker="o")
            ax.set_yscale("log")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("|f(x)|")
            ax.set_title(f"{method} convergence")
            ax.grid(True, alpha=0.3)
            show_plot(fig)

            st.download_button(
                "Download iteration data (CSV)",
                history_df.to_csv(index=False),
                file_name="root_finding_iterations.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Could not solve this problem: {exc}")

elif page == "Linear Systems":
    st.header("Linear System Solver")
    st.write("Solve a system of equations in matrix form: A x = b.")

    size = st.selectbox("Matrix size", [2, 3, 4], index=1)
    default_matrix = np.eye(size)
    default_rhs = np.ones(size)

    st.info("Enter coefficients row by row, separated by commas.")
    matrix_text = st.text_area(
        "Coefficient matrix A",
        value="\n".join(", ".join(str(float(v)) for v in row)
                        for row in default_matrix),
        height=120
    )
    rhs_text = st.text_input(
        "Right-hand side vector b",
        value=", ".join(str(float(v)) for v in default_rhs)
    )
    method = st.selectbox("Method", ["Gaussian Elimination", "LU Decomposition"])

    if st.button("Solve system", type="primary"):
        try:
            rows = [
                [float(value.strip()) for value in line.split(",")]
                for line in matrix_text.strip().splitlines() if line.strip()
            ]
            A = np.array(rows, dtype=float)
            b = np.array([float(value.strip()) for value in rhs_text.split(",")])

            if A.shape != (size, size) or b.shape != (size,):
                raise ValueError(f"Expected a {size}x{size} matrix and a vector of length {size}.")
            if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
                raise ValueError("Please enter finite numeric values.")

            if method == "Gaussian Elimination":
                x = gauss_solve(A, b)
            else:
                P, L, U = lu_decompose(A)
                x = lu_solve(P, L, U, b)

            residual = np.linalg.norm(A @ x - b)
            st.success("System solved.")
            c1, c2 = st.columns(2)
            c1.metric("Residual norm ||Ax-b||", f"{residual:.3e}")
            c2.metric("Solution dimension", len(x))
            result_df = pd.DataFrame({
                "Variable": [f"x{i+1}" for i in range(size)],
                "Value": x
            })
            st.dataframe(result_df, use_container_width=True)

            with st.expander("Show matrix details"):
                st.write("A", A)
                st.write("b", b)
                if method == "LU Decomposition":
                    st.write("Permutation matrix P", P)
                    st.write("Lower triangular matrix L", L)
                    st.write("Upper triangular matrix U", U)
                    st.caption("The decomposition satisfies P · A = L · U.")

            st.download_button(
                "Download solution (CSV)",
                result_df.to_csv(index=False),
                file_name="linear_system_solution.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Could not solve this system: {exc}")

elif page == "ODE Solver":
    st.header("Ordinary Differential Equation Solver")
    st.write("Solve a first-order initial value problem y' = f(t, y).")
    st.latex(r"\frac{dy}{dt}=f(t,y), \qquad y(t_0)=y_0")
    example = st.selectbox(
        "Example problem",
        ["RC-style decay: y' = (5-y)", "Exponential decay: y' = -y"]
    )
    if example == "RC-style decay: y' = (5-y)":
        f = lambda t, y: 5-y
        exact = lambda t: 5*(1-np.exp(-np.asarray(t)))
        default_y0, default_end = 0.0, 5.0
    else:
        f = lambda t, y: -y
        exact = lambda t: np.exp(-np.asarray(t))
        default_y0, default_end = 1.0, 5.0

    c1, c2, c3, c4 = st.columns(4)
    method = c1.selectbox("Method", ["Euler", "RK4"])
    t0 = c2.number_input("Initial time", value=0.0)
    y0 = c3.number_input("Initial value", value=float(default_y0))
    h = c4.number_input("Step size (h)", min_value=0.0001, max_value=2.0,
                        value=0.1, format="%.4f")
    t_end = st.number_input("End time", min_value=0.0001, value=float(default_end))

    if st.button("Solve ODE", type="primary"):
        try:
            if t_end <= t0:
                raise ValueError("End time must be greater than initial time.")
            steps = (t_end-t0)/h
            if abs(steps-round(steps)) > 1e-9:
                raise ValueError("(End time - initial time) must be an exact multiple of h.")
            solver = euler if method == "Euler" else rk4
            t, y = solver(f, t0, y0, h, t_end)
            y_exact = exact(t)
            errors = np.abs(y-y_exact)
            st.success("ODE solved.")
            c1, c2 = st.columns(2)
            c1.metric("Maximum absolute error", f"{np.max(errors):.3e}")
            c2.metric("Final numerical value", f"{y[-1]:.8f}")

            result_df = pd.DataFrame({
                "t": t, "Numerical solution": y,
                "Exact solution": y_exact, "Absolute error": errors
            })
            st.subheader("Numerical results")
            st.dataframe(result_df, use_container_width=True)

            fig, ax = plt.subplots()
            ax.plot(t, y_exact, label="Exact solution", linewidth=2)
            ax.plot(t, y, "o--", label=method)
            ax.set_xlabel("Time")
            ax.set_ylabel("y(t)")
            ax.set_title("Numerical vs exact solution")
            ax.grid(True, alpha=0.3)
            ax.legend()
            show_plot(fig)

            st.download_button(
                "Download ODE results (CSV)",
                result_df.to_csv(index=False),
                file_name="ode_results.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Could not solve this ODE: {exc}")

else:
    st.header("RC Circuit Simulation")
    st.write("Simulate the charging response of a series RC circuit.")
    st.latex(r"RC\\frac{dV_c}{dt}+V_c=V_{in}")
    c1, c2, c3 = st.columns(3)
    R = c1.number_input("Resistance R (ohms)", min_value=1.0, value=1000.0)
    C = c2.number_input("Capacitance C (farads)", min_value=1e-9,
                        value=0.001, format="%.6f")
    Vin = c3.number_input("Input voltage Vin (V)", min_value=0.0, value=5.0)
    c1, c2, c3 = st.columns(3)
    h = c1.number_input("Step size h (seconds)", min_value=0.001,
                        value=0.1, format="%.3f")
    t_end = c2.number_input("Simulation duration (seconds)", min_value=0.1,
                            value=5.0)
    method = c3.selectbox("Numerical method", ["Euler", "RK4"])

    if st.button("Run simulation", type="primary"):
        try:
            f, exact, tau = rc_model(R=R, C=C, Vin=Vin)
            if abs(t_end/h-round(t_end/h)) > 1e-9:
                raise ValueError("Simulation duration must be an exact multiple of step size.")
            solver = euler if method == "Euler" else rk4
            t, y = solver(f, 0.0, 0.0, h, t_end)
            y_exact = exact(t)
            errors = np.abs(y-y_exact)
            result_df = pd.DataFrame({
                "Time (s)": t,
                "Numerical voltage (V)": y,
                "Analytical voltage (V)": y_exact,
                "Absolute error (V)": errors
            })

            st.success("Simulation completed.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Time constant τ = RC", f"{tau:.4g} s")
            c2.metric("Final capacitor voltage", f"{y[-1]:.5f} V")
            c3.metric("Maximum absolute error", f"{np.max(errors):.3e} V")

            fig, ax = plt.subplots()
            ax.plot(t, y_exact, label="Analytical solution", linewidth=2)
            ax.plot(t, y, "o--", label=method)
            ax.axhline(Vin, linestyle=":", label="Input voltage")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Capacitor voltage (V)")
            ax.set_title("RC circuit charging response")
            ax.grid(True, alpha=0.3)
            ax.legend()
            show_plot(fig)

            st.subheader("Simulation data")
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download simulation data (CSV)",
                result_df.to_csv(index=False),
                file_name="rc_circuit_results.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

st.divider()
st.caption("Numerical Methods Toolkit | B.Tech Project")
