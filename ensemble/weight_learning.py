import numpy as np
from sklearn.metrics import log_loss
from scipy.optimize import minimize

def _softmax(w):
    w = w - np.max(w); e = np.exp(w); return e / e.sum()

def grad_weight_fusion(proba_list, y_val):
    P = np.stack(proba_list, axis=0)  # (M,N,C)
    def obj(w):
        w = _softmax(w)
        proba = np.tensordot(w, P, axes=(0,0))
        return log_loss(y_val, proba)
    res = minimize(obj, x0=np.zeros(P.shape[0]), method="L-BFGS-B")
    w = _softmax(res.x)
    proba = np.tensordot(w, P, axes=(0,0))
    return proba, w
