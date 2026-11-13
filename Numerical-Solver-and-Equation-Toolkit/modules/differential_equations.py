import numpy as np, sympy as sp
def _rhs(s):
 t,y=sp.symbols("t y")
 try:e=sp.sympify(s,locals={"t":t,"y":y,"sin":sp.sin,"cos":sp.cos,"exp":sp.exp,"log":sp.log,"sqrt":sp.sqrt})
 except Exception as ex:raise ValueError(f"Invalid ODE: {ex}")
 if e.free_symbols-{t,y}:raise ValueError("Use only t and y.")
 fn=sp.lambdify((t,y),e,"numpy")
 def f(a,b):
  z=float(fn(a,b))
  if not np.isfinite(z):raise ValueError("ODE returned non-finite value.")
  return z
 return f
def solve_ode(rhs,y0,t0,tf,h,method="RK4"):
 if h<=0 or tf<=t0:raise ValueError("Require h > 0 and tf > t0.")
 f=_rhs(rhs);n=int(np.ceil((tf-t0)/h));t=np.linspace(t0,tf,n+1);y=np.empty(n+1);y[0]=y0
 for i in range(n):
  dt=t[i+1]-t[i];a,b=t[i],y[i];k1=f(a,b)
  if method=="Euler":y[i+1]=b+dt*k1
  elif method=="Modified Euler":
   k2=f(a+dt,b+dt*k1);y[i+1]=b+dt*(k1+k2)/2
  else:
   k2=f(a+dt/2,b+dt*k1/2);k3=f(a+dt/2,b+dt*k2/2);k4=f(a+dt,b+dt*k3);y[i+1]=b+dt*(k1+2*k2+2*k3+k4)/6
 return t,y
