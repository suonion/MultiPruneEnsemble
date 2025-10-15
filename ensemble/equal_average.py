import numpy as np

def equal_avg(proba_list):
    # proba_list: [ (N, C), ... ]
    return np.mean(np.stack(proba_list, axis=0), axis=0)
