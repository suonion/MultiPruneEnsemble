# methods/epic_method.py
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

def _disagree_rate(p1_labels, p2_labels):
    return np.mean(p1_labels != p2_labels)

def run_epic(
    X_train, y_train,
    X_prune, y_prune,
    X_test,
    K: int = 16,
    n_estimators: int = 128,
    seed: int = 42,
    d_max=None,
    n_min: int = 1,
    top_multiplier: int = 3,     
    min_div: float = 0.05        
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
    labels_p = [np.argmax(p, 1) for p in Pp]

    aucs = np.array([_macro_auc(Pp[i], y_prune) for i in range(len(ests))])
    order = np.argsort(-aucs)  
    M = max(K, min(len(ests), top_multiplier * K))
    cand = order[:M].tolist()

    selected: List[int] = []
    for i in cand:
        if len(selected) == 0:
            selected.append(i)
            if len(selected) >= K: break
            continue
        div_ok = np.mean([_disagree_rate(labels_p[i], labels_p[j]) for j in selected]) >= min_div
        if div_ok:
            selected.append(i)
            if len(selected) >= K: break

    if len(selected) < K:
        for i in cand:
            if i not in selected:
                selected.append(i)
                if len(selected) >= K:
                    break

    proba_test = np.mean([Pt[i] for i in selected], axis=0)
    return "EPIC", proba_test, selected
