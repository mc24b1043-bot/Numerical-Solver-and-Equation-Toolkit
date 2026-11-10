"""
Demo / experiments for the report. Run:  python demo.py
Prints all tables and saves plots in ./results/
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from toolkit import (bisection, newton, gauss_solve, lu_decompose, lu_solve,
                     euler, rk4, rc_model, solve)

os.makedirs("results", exist_ok=True)
BLUE, TEAL, GREY = "#0B5394", "#00897B", "#37474F"

# ------------------------------------------------ 1. ROOT FINDING  f(x)=x^2-2
print("=== 1. Root finding: f(x) = x^2 - 2 ===")
f = lambda x: x**2 - 2
df = lambda x: 2 * x
rb, kb, hb = bisection(f, 1, 2, tol=1e-6)
rn, kn, hn = newton(f, df, 1.5, tol=1e-12)
print(f"Bisection : root={rb:.8f}  iterations={kb}")
print(f"Newton    : root={rn:.12f}  iterations={kn}")
print("Newton iterates:", [round(v, 6) for v in hn[:4]])

# convergence plot
eb = [abs(c - np.sqrt(2)) for c in hb]
en = [abs(c - np.sqrt(2)) + 1e-17 for c in hn]
plt.figure(figsize=(5, 3.4))
plt.semilogy(range(1, len(eb) + 1), eb, "s--", color=BLUE, ms=3, label="Bisection")
plt.semilogy(range(0, len(en)), en, "^-", color=TEAL, ms=4, label="Newton-Raphson")
plt.xlabel("Iteration"); plt.ylabel("|error|"); plt.grid(alpha=.3); plt.legend()
plt.title("Root-finding convergence"); plt.tight_layout()
plt.savefig("results/root_convergence.png", dpi=200); plt.close()

# ------------------------------------------------ 2. LINEAR SYSTEMS
print("\n=== 2. Linear system ===")
A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
b = [8, -11, -3]
x_g = gauss_solve(A, b)
P, L, U = lu_decompose(A)
x_lu = lu_solve(P, L, U, b)
print("Gauss  :", x_g)
print("LU     :", x_lu)
print("NumPy  :", np.linalg.solve(A, b))
print("Check P@A == L@U :", np.allclose(P @ np.array(A, float), L @ U))
print("Residual (LU)    :", np.linalg.norm(np.array(A) @ x_lu - b))

# ------------------------------------------------ 3. RC CIRCUIT
print("\n=== 3. RC circuit: R=1k, C=1mF, Vin=5V, tau=1s, h=0.5 ===")
fode, exact, tau = rc_model(R=1e3, C=1e-3, Vin=5.0)
h, T = 0.5, 5.0
te, ye = euler(fode, 0, 0.0, h, T)
tr, yr = rk4(fode, 0, 0.0, h, T)
print(f"{'t':>4} {'Exact':>8} {'Euler':>8} {'RK4':>8} {'|e_Eul|':>9} {'|e_RK4|':>9}")
for i, t in enumerate(te):
    if abs(t - round(t)) < 1e-9 and t >= 1:
        ex = float(exact(t))
        print(f"{t:4.0f} {ex:8.4f} {ye[i]:8.4f} {yr[i]:8.4f} "
              f"{abs(ye[i]-ex):9.4f} {abs(yr[i]-ex):9.5f}")

# Fig. 3 : comparison plot
tt = np.linspace(0, T, 400)
plt.figure(figsize=(5.5, 3.8))
plt.plot(tt, exact(tt), color=GREY, lw=2.2, label=r"Analytical $5(1-e^{-t})$")
plt.plot(te, ye, "s--", color=BLUE, ms=4, label="Euler, h=0.5 s")
plt.plot(tr, yr, "^", color=TEAL, ms=6, label="RK4, h=0.5 s")
plt.axhline(5, color="gray", ls=":", lw=1)
plt.xlabel("Time t (s)"); plt.ylabel(r"$V_c(t)$ (V)")
plt.title("RC step response: analytical vs numerical")
plt.grid(alpha=.3); plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig("results/rc_comparison.png", dpi=200); plt.close()

# ------------------------------------------------ 4. ORDER OF ACCURACY (extra)
print("\n=== 4. Error vs step size at t = 5 s (shows O(h) and O(h^4)) ===")
hs = [1.0, 0.5, 0.25, 0.125, 0.0625]
err_e, err_r = [], []
for hh in hs:
    _, y1 = euler(fode, 0, 0.0, hh, T)
    _, y2 = rk4(fode, 0, 0.0, hh, T)
    ex = float(exact(T))
    err_e.append(abs(y1[-1] - ex)); err_r.append(abs(y2[-1] - ex))
    print(f"h={hh:7.4f}  Euler err={err_e[-1]:.3e}  RK4 err={err_r[-1]:.3e}")

plt.figure(figsize=(5, 3.6))
plt.loglog(hs, err_e, "s-", color=BLUE, label="Euler  (slope ~ 1)")
plt.loglog(hs, err_r, "^-", color=TEAL, label="RK4  (slope ~ 4)")
plt.xlabel("Step size h (s)"); plt.ylabel("Absolute error at t = 5 s")
plt.grid(alpha=.3, which="both"); plt.legend(); plt.title("Order of accuracy")
plt.tight_layout(); plt.savefig("results/order_of_accuracy.png", dpi=200); plt.close()

# ------------------------------------------------ 5. WORKFLOW DISPATCHER
print("\n=== 5. Dispatcher (Fig. 4 workflow) ===")
r = solve("root", f=f, df=df, x0=1.5)
print("root  ->", r["method"], round(r["root"], 10), "residual", r["residual"])
r = solve("linear", A=A, b=b)
print("linear->", r["method"], r["x"], "residual", r["residual"])
r = solve("ode", f=fode, t0=0, y0=0.0, h=0.5, t_end=5, method="rk4", exact=exact)
print("ode   ->", r["method"], "max error", round(r["max_error"], 6))
print("\nPlots saved in ./results/")
