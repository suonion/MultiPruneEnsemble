# methods/disc_method.py
import numpy as np
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier

def _tree_error_on_prune(tree, Xp, yp) -> float:
    yhat = tree.predict(Xp)
    return 1.0 - np.mean(yhat == yp)

def _pair_disagreement(ti_pred, tj_pred) -> float:
    return np.mean(ti_pred != tj_pred)

def _marginal_diversity(t_pred, selected_preds) -> float:
    if len(selected_preds) == 0:
        return 1.0
    return np.mean([_pair_disagreement(t_pred, sp) for sp in selected_preds])

def _disc_select(preds_on_prune: np.ndarray, y_prune: np.ndarray, K: int,
                 alpha: float = 0.5) -> List[int]:
    M = preds_on_prune.shape[0]
    errs = np.array([1.0 - np.mean(preds_on_prune[i] == y_prune) for i in range(M)])
    selected, remaining = [], set(range(M))
    selected_preds = []

    while len(selected) < K and remaining:
        best_i, best_score = None, -1e18
        for i in list(remaining):
            div_i = _marginal_diversity(preds_on_prune[i], selected_preds)
            score = -(alpha * errs[i]) + (1 - alpha) * div_i
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.remove(best_i)
        selected_preds.append(preds_on_prune[best_i])
    return selected

def _avg_estimators_proba(estimators, X) -> np.ndarray:
    P = [e.predict_proba(X) for e in estimators]
    return np.mean(P, axis=0)

def run_disc(
    X_train, y_train,
    X_prune, y_prune,
    X_test,
    K: int = 16,
    n_estimators: int = 128,
    seed: int = 42,
    alpha: float = 0.5,
    d_max=True,
    n_min: int = 1
) -> Tuple[str, np.ndarray, List[int]]:

    rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=d_max,
    min_samples_leaf=n_min,
    random_state=seed)

    rf.fit(X_train, y_train)
    ests = rf.estimators_

    preds_prune = np.vstack([e.predict(X_prune) for e in ests])  # (M, Np)
    selected_idx = _disc_select(preds_prune, y_prune, K=K, alpha=alpha)

    chosen = [ests[i] for i in selected_idx]
    proba_test = _avg_estimators_proba(chosen, X_test)
    return "DISC", proba_test, selected_idx
