import numpy as np
from scipy import signal
def analyze_transfer_function(num,den,duration=10,points=600):
 num=np.trim_zeros(np.asarray(num,float),"f");den=np.trim_zeros(np.asarray(den,float),"f")
 if not len(num) or len(den)<2 or not np.any(den):raise ValueError("Need nonzero numerator and denominator degree >= 1.")
 sys=signal.TransferFunction(num,den);p=np.roots(den);z=np.roots(num) if len(num)>1 else np.array([])
 t=np.linspace(0,duration,points);_,step=signal.step(sys,T=t);_,impulse=signal.impulse(sys,T=t)
 w,mag,phase=signal.bode(sys,w=np.logspace(-2,2,500))
 return dict(num=num,den=den,poles=p,zeros=z,stable=bool(np.all(p.real<0)),t=t,step=step,impulse=impulse,omega=w,magnitude=mag,phase=phase)
