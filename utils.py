import os
import sys
import math
import time
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def here(*paths) -> str:
    return os.path.join(os.getcwd(), *paths)

def file_dir(file: str) -> str:
    return os.path.dirname(os.path.abspath(file))

def seed_everything(seed: int = 50):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

def split_train_prune_test(
    X, y,
    test_size: float = 0.25,
    prune_size: float = 0.25,
    seed: int = 50
):

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_prune, y_train, y_prune = train_test_split(
        X_tmp, y_tmp, test_size=prune_size, stratify=y_tmp, random_state=seed
    )
    return X_train, y_train, X_prune, y_prune, X_test, y_test

def softmax(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = w - np.max(w)
    e = np.exp(w)
    return e / np.sum(e)

def avg_prob(proba_list: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:

    P = np.stack(proba_list, axis=0)  # (M,N,C)
    if weights is None:
        return P.mean(axis=0)
    w = softmax(np.asarray(weights))
    return np.tensordot(w, P, axes=(0, 0))  # (N,C)

def avg_estimators_proba(estimators, X) -> np.ndarray:
    P = [est.predict_proba(X) for est in estimators]
    return np.mean(P, axis=0)


def eval_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:

    y_pred = np.argmax(proba, axis=1)
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "f1":  f1_score(y_true, y_pred, average="macro"),
    }
    C = proba.shape[1]
    try:
        if C == 2:
            out["auc"] = roc_auc_score(y_true, proba[:, 1])
        else:
            out["auc"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    except Exception:

        out["auc"] = float("nan")
    return out


def timed_call(fn: Callable, *args, **kwargs):

    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    n = None
    try:

        if isinstance(res, np.ndarray) and res.ndim == 2:
            n = res.shape[0]
        elif "X" in kwargs and hasattr(kwargs["X"], "__len__"):
            n = len(kwargs["X"])
        elif len(args) > 0 and hasattr(args[0], "__len__"):
            n = len(args[0])
    except Exception:
        pass
    per_ms = (dt / n * 1e3) if n else None
    return res, dt, per_ms

def pretty_metrics_table(rows: List[Tuple[str, Dict[str, float]]]) -> str:
    """
    rows: [(name, {"acc":..., "f1":..., "auc":...}), ...]
    """
    lines = []
    head = f"{'Method':18s}  {'ACC':>8s}  {'F1':>8s}  {'AUC':>8s}"
    lines.append(head)
    lines.append("-" * len(head))
    for name, m in rows:
        acc = f"{m.get('acc', float('nan')):.4f}"
        f1  = f"{m.get('f1', float('nan')):.4f}"
        auc = m.get("auc", float('nan'))
        auc = "   n/a " if (auc is None or (isinstance(auc, float) and math.isnan(auc))) else f"{auc:.4f}"
        lines.append(f"{name:18s}  {acc:>8s}  {f1:>8s}  {auc:>8s}")
    return "\n".join(lines)

def make_tree(d_max=None, n_min=1, seed=0, use_limits=False):

    if not use_limits:
        return DecisionTreeClassifier(random_state=seed)
    else:
        return DecisionTreeClassifier(
            max_depth=d_max,
            min_samples_leaf=n_min,
            random_state=seed
        )
