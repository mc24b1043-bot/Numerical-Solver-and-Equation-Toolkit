import numpy as np
from scipy import signal
def make_signal(kind,amplitude,frequency,fs,duration,phase=0):
 if fs<=0 or duration<=0 or frequency<=0 or frequency>=fs/2:raise ValueError("Require fs > 2f and positive duration.")
 t=np.arange(0,duration,1/fs);w=2*np.pi*frequency*t+np.deg2rad(phase)
 return t, amplitude*(np.sin(w) if kind=="Sine" else np.cos(w) if kind=="Cosine" else signal.square(w))
def add_noise(y,std,seed=7):return np.asarray(y)+np.random.default_rng(seed).normal(0,std,len(y))
def spectrum(y,fs):
 n=len(y);a=2*np.abs(np.fft.rfft(y))/n;f=np.fft.rfftfreq(n,1/fs)
 if len(a):a[0]/=2
 if n%2==0 and len(a)>1:a[-1]/=2
 return f,a
def filter_signal(y,fs,cutoff,kind="Low-pass",order=4):
 if not 0<cutoff<fs/2:raise ValueError("Cutoff must be below Nyquist.")
 return signal.sosfiltfilt(signal.butter(order,cutoff,btype="lowpass" if kind=="Low-pass" else "highpass",fs=fs,output="sos"),y)
