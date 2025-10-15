# methods/auc_greedy_method.py
import numpy as np
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def _macro_auc_from_probs(proba: np.ndarray, y_true: np.ndarray) -> float:
    C = proba.shape[1]
    try:
        if C == 2:
            return roc_auc_score(y_true, proba[:, 1])
        else:
            return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    except Exception:
        return np.nan

def run_auc_greedy(
    X_train, y_train,
    X_prune, y_prune,
    X_test,
    K: int = 16,
    n_estimators: int = 128,
    seed: int = 42,
    d_max=None,
    n_min: int = 1
) -> Tuple[str, np.ndarray, List[int]]:
    rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=d_max,
    min_samples_leaf=n_min,
    random_state=seed)
    rf.fit(X_train, y_train)
    ests = rf.estimators_

    P_prune = [e.predict_proba(X_prune) for e in ests]  # list of (Np, C)
    P_test  = [e.predict_proba(X_test)  for e in ests]  # list of (Nt, C)

    selected: List[int] = []
    used = np.zeros(len(ests), dtype=bool)
    cur_sum = None
    cur_auc = -np.inf

    for _ in range(K):
        best_i, best_auc = None, -np.inf
        for i in range(len(ests)):
            if used[i]: 
                continue
            if cur_sum is None:
                avg_prune = P_prune[i]
            else:
                avg_prune = (cur_sum + P_prune[i]) / (len(selected) + 1)
            auc = _macro_auc_from_probs(avg_prune, y_prune)
            if np.isnan(auc):
                continue
            if auc > best_auc:
                best_auc, best_i = auc, i
        if best_i is None:
            break
        used[best_i] = True
        selected.append(best_i)
        cur_sum = P_prune[best_i] if cur_sum is None else (cur_sum + P_prune[best_i])
        cur_auc = best_auc

    if len(selected) == 0:
        single_aucs = [ _macro_auc_from_probs(P_prune[i], y_prune) for i in range(len(ests)) ]
        best_i = int(np.nanargmax(single_aucs))
        selected = [best_i]

    proba_test = np.mean([P_test[i] for i in selected], axis=0)
    return "AUC-Greedy", proba_test, selected
