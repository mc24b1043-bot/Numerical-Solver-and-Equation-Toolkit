import numpy as np
def errors(approx,exact):
 abs_err=np.abs(np.asarray(approx)-np.asarray(exact))
 rel=abs_err/np.maximum(np.abs(exact),1e-15)
 return abs_err,rel