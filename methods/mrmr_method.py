import numpy as np
from typing import List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score


def _mrmr_select(preds_on_prune: np.ndarray, y_prune: np.ndarray, K: int) -> List[int]:

    M = preds_on_prune.shape[0]
    relevance = np.array([
        mutual_info_score(preds_on_prune[i], y_prune) for i in range(M)
    ])
    selected = [int(np.argmax(relevance))]
    remaining = set(range(M)) - {selected[0]}
    mi_cache = {}

    def _pair_mi(i, j):
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in mi_cache:
            mi_cache[(a, b)] = mutual_info_score(preds_on_prune[a], preds_on_prune[b])
        return mi_cache[(a, b)]

    while len(selected) < K and remaining:
        best_i, best_score = None, -1e18
        for i in list(remaining):
            redundancy = np.mean([_pair_mi(i, j) for j in selected]) if selected else 0.0
            score = relevance[i] - redundancy
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.remove(best_i)
    return selected

def _avg_estimators_proba(estimators, X) -> np.ndarray:
    P = [est.predict_proba(X) for est in estimators]
    return np.mean(P, axis=0)

def run_mrmr(
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
    preds_prune = np.vstack([e.predict(X_prune) for e in ests])  # (M, N_prune)
    selected_idx = _mrmr_select(preds_prune, y_prune, K)

    chosen = [ests[i] for i in selected_idx]
    proba_test = _avg_estimators_proba(chosen, X_test)  # (N_test, C)
    return "MRMR", proba_test, selected_idx
