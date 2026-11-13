import numpy as np, sympy as sp
OPS={"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"exp":sp.exp,"log":sp.log,"sqrt":sp.sqrt,"abs":sp.Abs,"pi":sp.pi,"E":sp.E}
def parse_function(s,var="x"):
 if not s or len(s)>250: raise ValueError("Enter a valid expression (maximum 250 characters).")
 z=sp.Symbol(var)
 try:e=sp.sympify(s,locals={**OPS,var:z})
 except Exception as ex:raise ValueError(f"Invalid expression: {ex}")
 if e.free_symbols-{z}:raise ValueError(f"Only {var} is allowed.")
 fn=sp.lambdify(z,e,"numpy")
 def f(v):
  a=np.asarray(fn(v),float)
  if not np.all(np.isfinite(a)):raise ValueError("Function returned a non-finite value.")
  return float(a) if a.ndim==0 else a
 return e,f
def bisection(s,a,b,tol=1e-8,max_iter=100):
 _,f=parse_function(s); fa,fb=f(a),f(b)
 if fa*fb>0:raise ValueError("Bisection needs opposite signs at endpoints.")
 rows=[]
 for i in range(1,max_iter+1):
  c=(a+b)/2; fc=f(c); err=abs(b-a)/2; rows.append(dict(iteration=i,approximation=c,**{"f(x)":fc,"error":err}))
  if abs(fc)<tol or err<tol:break
  if fa*fc<0:b,fb=c,fc
  else:a,fa=c,fc
 return c,rows
def newton_raphson(s,x0,tol=1e-8,max_iter=100):
 e,f=parse_function(s); x=sp.Symbol("x"); d=sp.lambdify(x,sp.diff(e,x),"numpy"); rows=[]
 for i in range(1,max_iter+1):
  slope=float(d(x0))
  if abs(slope)<1e-14:raise ValueError("Derivative is zero.")
  z=x0-f(x0)/slope; err=abs(z-x0); rows.append({"iteration":i,"approximation":z,"f(x)":f(z),"error":err})
  if err<tol:return z,rows
  x0=z
 raise ValueError("Newton-Raphson did not converge.")
def secant(s,x0,x1,tol=1e-8,max_iter=100):
 _,f=parse_function(s); rows=[]
 for i in range(1,max_iter+1):
  f0,f1=f(x0),f(x1)
  if abs(f1-f0)<1e-14:raise ValueError("Secant denominator is zero.")
  z=x1-f1*(x1-x0)/(f1-f0); err=abs(z-x1); rows.append({"iteration":i,"approximation":z,"f(x)":f(z),"error":err})
  if err<tol:return z,rows
  x0,x1=x1,z
 raise ValueError("Secant did not converge.")
def gauss_elimination(A,b):
 A=np.array(A,float); b=np.array(b,float).reshape(-1); n=len(b)
 if A.shape!=(n,n):raise ValueError("A must be square and match b.")
 M=np.c_[A,b]
 for k in range(n):
  p=k+np.argmax(abs(M[k:,k]))
  if abs(M[p,k])<1e-12:raise ValueError("Matrix is singular.")
  M[[k,p]]=M[[p,k]]
  for i in range(k+1,n):M[i,k:]-=M[i,k]/M[k,k]*M[k,k:]
 out=np.zeros(n)
 for i in range(n-1,-1,-1):out[i]=(M[i,-1]-M[i,i+1:n]@out[i+1:n])/M[i,i]
 return out
def lu_decomposition(A,b):
 A=np.array(A,float); n=len(A)
 if A.shape!=(n,n):raise ValueError("A must be square.")
 L=np.eye(n); U=A.copy()
 for k in range(n):
  if abs(U[k,k])<1e-12:raise ValueError("Zero pivot in LU.")
  for i in range(k+1,n):L[i,k]=U[i,k]/U[k,k]; U[i,k:]-=L[i,k]*U[k,k:]
 return np.linalg.solve(U,np.linalg.solve(L,np.array(b,float))),L,U
def lagrange_interpolation(xs,ys,z):
 xs=np.array(xs,float);ys=np.array(ys,float)
 if len(xs)!=len(ys) or len(set(xs))!=len(xs):raise ValueError("x/y lengths must match and x values be unique.")
 ans=0.
 for i in range(len(xs)):
  term=ys[i]
  for j in range(len(xs)):
   if i!=j:term*= (z-xs[j])/(xs[i]-xs[j])
  ans+=term
 return float(ans)
def integrate(s,a,b,n=100,rule="Simpson"):
 _,f=parse_function(s); n=int(n)
 if n<1:raise ValueError("Intervals must be positive.")
 if rule=="Simpson":
  n+=n%2; h=(b-a)/n;y=np.asarray(f(np.linspace(a,b,n+1)))
  return float(h/3*(y[0]+y[-1]+4*y[1:-1:2].sum()+2*y[2:-1:2].sum())),n
 h=(b-a)/n;y=np.asarray(f(np.linspace(a,b,n+1)))
 return float(h*(y[0]/2+y[1:-1].sum()+y[-1]/2)),n
def differentiate(s,z,h=1e-4,method="Central"):
 _,f=parse_function(s)
 if h<=0:raise ValueError("Step must be positive.")
 if method=="Forward":return (f(z+h)-f(z))/h
 if method=="Backward":return (f(z)-f(z-h))/h
 return (f(z+h)-f(z-h))/(2*h)
