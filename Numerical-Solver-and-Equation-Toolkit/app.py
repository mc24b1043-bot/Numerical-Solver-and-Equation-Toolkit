import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd,plotly.graph_objects as go,streamlit as st,sympy as sp
from modules import numerical_analysis as n
from modules.differential_equations import solve_ode
from modules.linear_systems import analyze_transfer_function
from modules.signal_processing import make_signal,add_noise,spectrum,filter_signal,combine_signals,convolve_signals
from modules.system_simulator import mass_spring_damper,rlc_response,first_order
from utils.validation import parse_vector
from utils.pdf_generator import make_report
st.set_page_config(page_title="Numerical Solver & Equation Toolkit",page_icon="^2",layout="wide")
st.markdown("<style>.stApp{background:radial-gradient(ellipse at 15% 0%,#10283e,#0b1220 48%,#080d17);color:#e7eef8}[data-testid=stSidebar]{background:#0b1421}.block-container{padding-top:2rem;max-width:1450px}div[data-testid=stMetric]{background:#111e2d;border:1px solid #24384b;padding:14px;border-radius:14px}</style>",unsafe_allow_html=True)
PAGES=["Dashboard","Numerical Analysis","Linear Systems","Signal Processing","Differential Equations","System Simulator","Intelligent Analyzer","About Project"]
with st.sidebar:
 st.image(ROOT / "assets" / "logo.png", width=72, alt="Toolkit logo");st.markdown("## NS&T");st.caption("SCIENTIFIC COMPUTING SUITE");page=st.radio("Navigation",PAGES,label_visibility="collapsed");st.divider();st.caption("Numerical analysis | signals and systems")
def header(num,title,desc):
 st.caption(num.upper());st.title(title);st.write(desc);st.divider()
def chart(df,x,ys,title):
 f=go.Figure()
 for y in ys:f.add_trace(go.Scatter(x=df[x],y=df[y],name=y,mode="lines"))
 f.update_layout(template="plotly_dark",title=title,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",legend=dict(orientation="h"))
 st.plotly_chart(f,width="stretch",alt=title)
def pdf(title,**kw):st.download_button("Download PDF report",make_report(title,kw),"calculation_report.pdf","application/pdf")
def dashboard():
 header("Integrated engineering workspace","Numerical Solver & Equation Toolkit","A practical bridge between numerical analysis, linear systems and signal processing.")
 cols=st.columns(4)
 for c,(label,val) in zip(cols,[("Numerical methods","12"),("Engineering models","3"),("System diagnostics","4"),("Signal tools","8")]):c.metric(label,val)
 st.subheader("Explore the toolkit")
 for row in [PAGES[1:4],PAGES[4:7]]:
  cols=st.columns(3)
  for c,p in zip(cols,row):
   with c,st.container(border=True):st.subheader(p);st.write({"Numerical Analysis":"Roots, linear algebra, interpolation and calculus.","Linear Systems":"Transfer functions, stability and Bode responses.","Signal Processing":"Waveforms, noise, filtering and FFT.","Differential Equations":"Compare Euler, Heun and RK4.","System Simulator":"Mechanical, electrical and first-order models.","Intelligent Analyzer":"Rule-based recommendations and an RLC workflow."}[p])
 st.info("Quick start: solve dy/dt = -2y, y(0)=1 in Differential Equations.")
def numerical():
 header("Module 01","Numerical analysis","Select a method; inspect iteration tables and computed values.")
 tabs=st.tabs(["Root finding","Linear equations","Interpolation","Integration","Differentiation"])
 with tabs[0]:
  expr=st.text_input("f(x)","x**3-x-2");method=st.selectbox("Method",["Bisection","Newton-Raphson","Secant"]);tol=st.number_input("Tolerance",value=1e-7,format="%.1e")
  if method=="Bisection":a=st.number_input("a",value=1.);b=st.number_input("b",value=2.)
  elif method=="Newton-Raphson":a=st.number_input("Initial guess",value=1.5);b=a
  else:a=st.number_input("First guess",value=1.);b=st.number_input("Second guess",value=2.)
  if st.button("Solve root",type="primary"):
   try:
    ans,rows={"Bisection":n.bisection,"Newton-Raphson":n.newton_raphson,"Secant":n.secant}[method](expr,a,b,tol);st.success(f"Root {ans:.10g}, f(root)={n.parse_function(expr)[1](ans):.3g}");st.dataframe(pd.DataFrame(rows),hide_index=True,alt="Root iterations")
    _,f=n.parse_function(expr);x=np.linspace(a-1,b+1,250);fig=go.Figure(go.Scatter(x=x,y=f(x)));fig.add_hline(y=0);fig.add_vline(x=ans);fig.update_layout(template="plotly_dark",title="Root convergence");st.plotly_chart(fig,width="stretch",alt="Function root graph");pdf("Root finding",Function=expr,Method=method,Root=ans,Iterations=len(rows),Error=rows[-1]["error"])
   except Exception as e:st.error(str(e))
 with tabs[1]:
  A=st.text_area("Matrix A","[[2,1],[1,3]]");b=st.text_input("Vector b","[5,6]");meth=st.selectbox("Method",["Gauss elimination","LU decomposition"])
  if st.button("Solve system"):
   try:
    A=np.array(__import__("ast").literal_eval(A),float);b=np.array(__import__("ast").literal_eval(b),float);z=n.gauss_elimination(A,b) if meth.startswith("Gauss") else n.lu_decomposition(A,b)[0];st.success(f"x = {z}");st.metric("Residual",f"{np.linalg.norm(A@z-b):.2e}");pdf("Linear system",Method=meth,Solution=z.tolist(),Residual=np.linalg.norm(A@z-b))
   except Exception as e:st.error(str(e))
 with tabs[2]:
  xs=st.text_input("x data","0,1,2");ys=st.text_input("y data","1,3,2");z=st.number_input("Evaluate at x",value=1.5)
  if st.button("Interpolate"):
   try:
    X=parse_vector(xs);Y=parse_vector(ys);v=n.lagrange_interpolation(X,Y,z);st.metric("Interpolated value",f"{v:.10g}");xx=np.linspace(min(X),max(X),200);chart(pd.DataFrame({"x":xx,"Polynomial":[n.lagrange_interpolation(X,Y,q) for q in xx]}),"x",["Polynomial"],"Lagrange polynomial");pdf("Interpolation",X=X,Y=Y,Point=z,Value=v)
   except Exception as e:st.error(str(e))
 with tabs[3]:
  expr=st.text_input("Integrand","x**2");a=st.number_input("Lower limit",value=0.);b=st.number_input("Upper limit",value=2.);count=st.number_input("Intervals",1,20000,100);rule=st.selectbox("Rule",["Trapezoidal","Simpson"])
  if st.button("Integrate"):
   try:
    value,nn=n.integrate(expr,a,b,count,rule);ex=sp.integrate(sp.sympify(expr),(sp.Symbol("x"),a,b));st.metric("Approximate integral",f"{value:.12g}");st.caption(f"{rule}, {nn} subintervals"+(f"; exact {float(ex):.12g}, error {abs(float(ex)-value):.3e}" if ex.is_number else ""));pdf("Integration",Function=expr,Bounds=[a,b],Rule=rule,Result=value)
   except Exception as e:st.error(str(e))
 with tabs[4]:
  expr=st.text_input("Function","sin(x)");z=st.number_input("Point",value=1.);h=st.number_input("Step",value=.001);method=st.selectbox("Difference",["Forward","Backward","Central"])
  if st.button("Differentiate"):
   try:
    v=n.differentiate(expr,z,h,method);st.metric("Derivative approximation",f"{v:.12g}");st.caption(f"Symbolic reference: {float(sp.diff(n.parse_function(expr)[0],sp.Symbol('x')).subs('x',z)):.12g}");pdf("Differentiation",Function=expr,Point=z,Step=h,Method=method,Value=v)
   except Exception as e:st.error(str(e))
def linear():
 header("Module 02","Linear systems","Transfer function poles, zeros, stability, time and frequency responses.")
 with st.form("tf"):
  num=st.text_input("Numerator coefficients, highest power first","1");den=st.text_input("Denominator coefficients","1,3,2");dur=st.number_input("Duration",1.,100.,10.);run=st.form_submit_button("Analyze",type="primary")
 if run:
  try:
   r=analyze_transfer_function(parse_vector(num),parse_vector(den),dur);st.latex(r"G(s)=\frac{"+",".join(map(str,r["num"]))+r"}{"+",".join(map(str,r["den"]))+r"}");cols=st.columns(3);cols[0].metric("Poles",str(np.round(r["poles"],5)));cols[1].metric("Zeros",str(np.round(r["zeros"],5)));cols[2].metric("Stability","Stable" if r["stable"] else "Unstable")
   chart(pd.DataFrame({"Time":r["t"],"Step":r["step"],"Impulse":r["impulse"]}),"Time",["Step","Impulse"],"Time response")
   for key,title in [("magnitude","Bode magnitude (dB)"),("phase","Bode phase (degrees)")]:
    f=go.Figure(go.Scatter(x=r["omega"],y=r[key]));f.update_xaxes(type="log",title="rad/s");f.update_layout(template="plotly_dark",title=title);st.plotly_chart(f,width="stretch",alt=title)
   f=go.Figure();f.add_hline(y=0);f.add_vline(x=0)
   for key,label,symbol in [("poles","Poles","x"),("zeros","Zeros","circle")]:
    q=r[key];f.add_trace(go.Scatter(x=q.real,y=q.imag,mode="markers",name=label,marker_symbol=symbol,marker_size=12))
   f.update_layout(template="plotly_dark",title="Pole-zero map",xaxis_title="Real",yaxis_title="Imaginary");st.plotly_chart(f,width="stretch",alt="Pole-zero map");pdf("Linear systems",Numerator=r["num"].tolist(),Denominator=r["den"].tolist(),Poles=r["poles"].tolist(),Zeros=r["zeros"].tolist(),Stable=r["stable"])
  except Exception as e:st.error(str(e))
def signals():
 header("Module 03","Signal processing","Generate a signal and compare original, noisy, filtered and frequency-domain views.")
 a,b,c=st.columns(3);wave=a.selectbox("Waveform",["Sine","Cosine","Square"]);amp=a.number_input("Amplitude",value=1.);freq=b.number_input("Frequency Hz",value=5.,min_value=.01);fs=b.number_input("Sampling Hz",value=200.,min_value=1.);phase=c.number_input("Phase degrees",value=0.);noise=c.number_input("Noise std",value=.25,min_value=0.);cut=c.number_input("Cutoff Hz",value=12.,min_value=.01);ft=st.selectbox("Filter",["Low-pass","High-pass"])
 if st.button("Process signal",type="primary"):
  try:
   t,y=make_signal(wave,amp,freq,fs,2,phase);yn=add_noise(y,noise);yf=filter_signal(yn,fs,cut,ft);xx,sp0=spectrum(y,fs);_,sp1=spectrum(yn,fs);chart(pd.DataFrame({"Time":t,"Original":y,"Noisy":yn,"Filtered":yf}),"Time",["Original","Noisy","Filtered"],"Time domain");chart(pd.DataFrame({"Frequency":xx,"Original":sp0,"Noisy":sp1}),"Frequency",["Original","Noisy"],"One-sided FFT amplitude");st.info("FFT maps sampled data to frequency bins. Butterworth filters provide a smooth passband; forward-backward filtering avoids phase delay.");pdf("Signal processing",Waveform=wave,Amplitude=amp,Frequency=freq,Sampling=fs,Noise=noise,Filter=ft,Cutoff=cut)
  except Exception as e:st.error(str(e))
 with st.expander("Signal addition, multiplication and convolution"):
  f2=st.number_input("Second signal frequency (Hz)",value=8.0,min_value=.01,key="sig_f2")
  operation=st.selectbox("Pointwise operation",["Addition","Multiplication"],key="sig_op")
  if st.button("Compute signal operation",key="sig_compute"):
   try:
    _,y1=make_signal(wave,amp,freq,fs,2,phase);_,y2=make_signal(wave,amp,f2,fs,2,phase)
    combined=combine_signals(y1,y2,operation);conv=convolve_signals(y1[:min(100,len(y1))],y2[:min(100,len(y2))])
    chart(pd.DataFrame({"Time":t,"Combined":combined}),"Time",["Combined"],f"Pointwise {operation.lower()}")
    chart(pd.DataFrame({"Sample index":np.arange(len(conv)),"Convolution":conv}),"Sample index",["Convolution"],"Discrete linear convolution")
   except Exception as e:st.error(str(e))

def differential():
 header("Module 04","Differential equations","Compare Euler, modified Euler (Heun) and fourth-order Runge-Kutta.")
 a,b=st.columns(2);rhs=a.text_input("dy/dt =","-2*y");y0=a.number_input("Initial y",value=1.);exact=a.text_input("Exact y(t), optional","exp(-2*t)");t0=b.number_input("t",value=0.);tf=b.number_input("t final",value=2.);h=b.number_input("Step size",value=.2,min_value=.001)
 if st.button("Compare methods",type="primary"):
  try:
   out={m:solve_ode(rhs,y0,t0,tf,h,m) for m in ["Euler","Modified Euler","RK4"]};t=out["RK4"][0];d={"t":t,**{m:v[1] for m,v in out.items()}}
   try:_,f=n.parse_function(exact,"t");d["Exact"]=f(t)
   except Exception:pass
   df=pd.DataFrame(d);chart(df,"t",[x for x in df.columns if x!="t"],"Numerical solution comparison")
   if "Exact" in df:
    for m in out:df[m+" abs error"]=abs(df[m]-df["Exact"])
    st.metric("Final RK4 absolute error",f'{df["RK4 abs error"].iloc[-1]:.3e}')
   st.dataframe(df,hide_index=True,alt="ODE solutions and errors");pdf("ODE solution",Equation=rhs,Initial=y0,Interval=[t0,tf],Step=h,FinalEuler=out["Euler"][1][-1],FinalHeun=out["Modified Euler"][1][-1],FinalRK4=out["RK4"][1][-1])
  except Exception as e:st.error(str(e))
def simulator():
 header("Module 05","System simulator","Explore mechanical, electrical and first-order dynamic models.")
 model=st.selectbox("Model",["Mass-spring-damper","Series RLC circuit","First-order system"])
 if model=="Mass-spring-damper":
  a,b,c=st.columns(3);m=a.number_input("Mass kg",value=1.,min_value=.01);damp=a.number_input("Damping",value=1.,min_value=0.);k=b.number_input("Stiffness",value=4.,min_value=.01);F=b.number_input("Force",value=1.);x0=c.number_input("Initial displacement",value=0.);v0=c.number_input("Initial velocity",value=0.);dur=c.number_input("Duration s",value=10.)
  if st.button("Simulate"):
   try:
    r=mass_spring_damper(m,damp,k,F,x0,v0,dur);chart(pd.DataFrame({"t":r["t"],"Displacement":r["position"],"Velocity":r["velocity"],"Acceleration":r["acceleration"]}),"t",["Displacement","Velocity","Acceleration"],"Mechanical response");st.metric("Damping ratio#",f'{r["zeta"]:.4f}');st.write("Poles:",r["poles"]);pdf(model,m=m,c=damp,k=k,force=F,zeta=r["zeta"],poles=r["poles"].tolist())
   except Exception as e:st.error(str(e))
 elif model=="Series RLC circuit":
  a,b,c=st.columns(3);R=a.number_input("Resistance ",value=2.,min_value=.01);L=b.number_input("Inductance H",value=1.,min_value=.01);C=c.number_input("Capacitance F",value=.5,min_value=.001);V=a.number_input("Voltage step",value=1.);dur=b.number_input("Duration s",value=10.)
  if st.button("Simulate"):
   try:
    r=rlc_response(R,L,C,V,dur);chart(pd.DataFrame({"t":r["t"],"Charge":r["charge"],"Current":r["current"]}),"t",["Charge","Current"],"RLC response");st.code(f"Q(s)/V(s) = 1 / ({L:g}s + {R:g}s + {1/C:g})");st.write("Poles:",r["poles"]);pdf(model,R=R,L=L,C=C,V=V,poles=r["poles"].tolist())
   except Exception as e:st.error(str(e))
 else:
  a,b,c=st.columns(3);tau=a.number_input("Time constant ",value=1.,min_value=.01);K=b.number_input("Gain K",value=2.);u=c.number_input("Input amplitude",value=1.)
  if st.button("Simulate"):
   t,y=first_order(tau,K,u,5*tau);chart(pd.DataFrame({"t":t,"Output":y}),"t",["Output"],"First-order step");st.caption("At one time constant, output reaches 63.2% of its final value.");pdf(model,tau=tau,gain=K,input=u,final=K*u)
def analyzer():
 header("Integrated analyzer","Intelligent Numerical and System Analyzer","Transparent rule-based recommendations connect the math method to the system model.")
 cat=st.selectbox("Problem type",["Scalar root","Initial-value ODE","Transfer function","RLC circuit"])
 advice={"Scalar root":"Bisection guarantees convergence for a bracket with opposite endpoint signs. Newton is fast near a root when the derivative is nonzero.","Initial-value ODE":"RK4 is a strong default for smooth, non-stiff problems. Compare Euler and Heun to see accuracy gains.","Transfer function":"Start with denominator poles; all real parts must be negative for continuous-time asymptotic stability.","RLC circuit":"Use Lq''+Rq'+q/C=V(t), then inspect roots of Ls+Rs+1/C and compare with computed response."}
 st.info(advice[cat],title="Recommended approach");st.markdown("#### Integrated RLC workflow");st.write("Parameters! differential equation! numerical response! transfer function  poles! stability! PDF")
 R=st.number_input("R ()",value=2.,min_value=.01,key="ir");L=st.number_input("L (H)",value=1.,min_value=.01,key="il");C=st.number_input("C (F)",value=.5,min_value=.001,key="ic");V=st.number_input("Input step (V)",value=1.,key="iv")
 if st.button("Run integrated RLC analysis",type="primary"):
  r=rlc_response(R,L,C,V);stable=bool(np.all(r["poles"].real<0));st.latex(rf"{L}\ddot q+{R}\dot q+{1/C}q={V}u(t)");st.metric("Stability","Stable" if stable else "Unstable");st.write("Q(s)/V(s) =",f"{V}/({L}s+{R}s+{1/C})","Poles:",r["poles"]);chart(pd.DataFrame({"t":r["t"],"Charge":r["charge"],"Current":r["current"]}),"t",["Charge","Current"],"Integrated RLC response");pdf("Integrated RLC analysis",R=R,L=L,C=C,V=V,ODE=f"{L}q''+{R}q'+{1/C}q={V}u(t)",Poles=r["poles"].tolist(),Stable=stable,FinalCharge=r["charge"][-1])
def about():
 header("Project guide","About this project","B.Tech major project connecting Numerical Analysis and Linear Systems & Signal Processing.")
 st.subheader("Subject integration");st.write("Numerical Analysis provides algorithms for roots, integrals, derivatives, linear equations and ODE trajectories. Signals and Systems provides dynamic models, transfer functions, frequency response and sampled data. Here a physical model becomes a differential equation; a numerical solver computes its trajectory; pole analysis independently explains stability.")
 st.subheader("Viva explanation");st.info("For a series RLC circuit, Kirchhoff's voltage law gives Lq''+Rq'+q/C=V(t). Its transfer function is Q(s)/V(s)=1/(Ls+Rs+1/C). Denominator roots are poles. The app computes charge/current curves and checks stability from pole real parts. The analyzer uses explicit rules, not machine learning.")
 st.write("Limitations: the scalar ODE solver expects explicit dy/dt=f(t,y). Root methods depend on brackets or initial values. System responses assume continuous-time LTI models. Inputs and units are the user's responsibility.")
pagefn={"Dashboard":dashboard,"Numerical Analysis":numerical,"Linear Systems":linear,"Signal Processing":signals,"Differential Equations":differential,"System Simulator":simulator,"Intelligent Analyzer":analyzer,"About Project":about}
pagefn[page]()
