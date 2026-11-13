import numpy as np
from scipy.integrate import solve_ivp
def mass_spring_damper(m,c,k,force,x0,v0,duration=10,points=700):
 if m<=0 or k<=0 or c<0:raise ValueError("Require m,k > 0 and c >= 0.")
 t=np.linspace(0,duration,points);F=float(force)
 z=solve_ivp(lambda _,z:[z[1],(F-c*z[1]-k*z[0])/m],(0,duration),[x0,v0],t_eval=t,rtol=1e-8,atol=1e-10)
 x,v=z.y;p=np.roots([m,c,k])
 return dict(t=t,position=x,velocity=v,acceleration=(F-c*v-k*x)/m,zeta=c/(2*np.sqrt(m*k)),poles=p)
def rlc_response(R,L,C,V,duration=10,points=700):
 if min(R,L,C)<=0:raise ValueError("R, L and C must be positive.")
 from scipy.signal import TransferFunction,step
 t=np.linspace(0,duration,points);den=[L,R,1/C];_,q=step(TransferFunction([V],den),T=t);p=np.roots(den)
 return dict(t=t,charge=q,current=np.gradient(q,t),poles=p,num=[V],den=den)
def first_order(tau,gain,amplitude,duration=10,points=500):
 if tau<=0:raise ValueError("Time constant must be positive.")
 t=np.linspace(0,duration,points);return t,gain*amplitude*(1-np.exp(-t/tau))
