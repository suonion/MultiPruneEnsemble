# methods/hybrid_method.py
import numpy as np
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def _macro_auc(proba, y):
    C = proba.shape[1]
    try:
        if C == 2: return roc_auc_score(y, proba[:,1])
        return roc_auc_score(y, proba, multi_class="ovr", average="macro")
    except Exception:
        return -np.inf

def _avg_margin(proba: np.ndarray) -> float:
    part = np.sort(proba, axis=1)
    return np.mean(part[:, -1] - part[:, -2])

def _disagree_rate(y1: np.ndarray, y2: np.ndarray) -> float:
    return np.mean(y1 != y2)

def run_hybrid(
    X_train, y_train,
    X_prune, y_prune,
    X_test,
    K: int = 16,
    n_estimators: int = 128,
    seed: int = 42,
    d_max=None,
    n_min: int = 1,
    w_epic: float = 0.5,  
    w_umep: float = 0.5,    
    beta_div: float = 0.2   
) -> Tuple[str, np.ndarray, List[int]]:
    rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=d_max,
    min_samples_leaf=n_min,
    random_state=seed)
    rf.fit(X_train, y_train)
    ests = rf.estimators_

    Pp = [e.predict_proba(X_prune) for e in ests]
    Pt = [e.predict_proba(X_test)  for e in ests]
    yp_hat = [np.argmax(p, 1) for p in Pp]
    auc_i = np.array([_macro_auc(Pp[i], y_prune) for i in range(len(ests))])  # EPIC基分

    selected: List[int] = []
    used = np.zeros(len(ests), dtype=bool)
    cur_sum = None
    cur_margin = -np.inf

    for _ in range(K):
        best_i, best_score = None, -1e18
        for i in range(len(ests)):
            if used[i]:
                continue
            score_epic = auc_i[i]

            if cur_sum is None:
                avg_p = Pp[i]
                base_margin = -np.inf
                div = 0.0
            else:
                avg_p = (cur_sum + Pp[i]) / (len(selected) + 1)
                base_margin = cur_margin
                div = np.mean([_disagree_rate(yp_hat[i], yp_hat[j]) for j in selected])
            margin_gain = _avg_margin(avg_p) - (0 if base_margin == -np.inf else base_margin)
            score_umep = margin_gain + beta_div * div

            score = w_epic * score_epic + w_umep * score_umep
            if score > best_score:
                best_score, best_i = score, i

        if best_i is None:
            break
        used[best_i] = True
        selected.append(best_i)
        cur_sum = Pp[best_i] if cur_sum is None else (cur_sum + Pp[best_i])
        cur_margin = _avg_margin(cur_sum / len(selected))

    if len(selected) == 0:
        selected = [int(np.nanargmax(auc_i))]

    proba_test = np.mean([Pt[i] for i in selected], axis=0)
    return "Hybrid", proba_test, selected
