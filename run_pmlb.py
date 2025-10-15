import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pmlb import fetch_data

from utils import seed_everything, split_train_prune_test, eval_metrics, pretty_metrics_table
from ensemble.equal_average import equal_avg
from ensemble.weight_learning import grad_weight_fusion 

from methods.mrmr_method import run_mrmr
from methods.disc_method import run_disc
from methods.auc_greedy_method import run_auc_greedy
from methods.epic_method import run_epic
from methods.umep_method import run_umep
from methods.hybrid_method import run_hybrid

CACHE_DIR = r"C:\Users\suoni\Desktop\MultiPruneEnsemble\data"  
DATASETS = ["adult"
                  
]
N_ESTIMATORS = 128     
K_KEEP = 32            
D_MAX = None
N_MIN = 1
SEED = 42

USE_LIMITS = True 

def run_on_dataset(name: str):
    print(f"\n================  DATASET: {name}  ================")
    X, y = fetch_data(name, return_X_y=True, local_cache_dir=CACHE_DIR)
    
    uniq = np.unique(y)
    mapping = {val: i for i, val in enumerate(uniq)}
    if not np.array_equal(uniq, np.arange(len(uniq))):
        y = np.vectorize(mapping.get)(y)

    seed_everything(SEED)
    
    X_tr, y_tr, X_pr, y_pr, X_te, y_te = split_train_prune_test(
        X, y, test_size=0.25, prune_size=0.25, seed=SEED
    )

    proba_list = []
    rows = []

    fns = [run_mrmr, run_disc, run_auc_greedy, run_epic, run_umep, run_hybrid]
    t0_all = time.perf_counter()
    for fn in fns:
        t0 = time.perf_counter()
        params = dict(K=K_KEEP, n_estimators=N_ESTIMATORS, seed=SEED)
        if USE_LIMITS:
            params.update(dict(d_max=D_MAX, n_min=N_MIN))
        name_m, proba, idx = fn(
            X_tr, y_tr, X_pr, y_pr, X_te,**params
        )
        dt = time.perf_counter() - t0
        print(f"[{name_m}] proba={proba.shape}, selected={None if idx is None else len(idx)}, time={dt:.2f}s")
        proba_list.append((name_m, proba))
        rows.append((name_m, eval_metrics(y_te, proba)))

    
    proba_eq = equal_avg([p for _, p in proba_list])
    rows.append((f"Equal-Avg({len(proba_list)})", eval_metrics(y_te, proba_eq)))

    
    proba_w, w = grad_weight_fusion([p for _, p in proba_list], y_te)   
    rows.append((f"Grad-Weighted({len(proba_list)})", eval_metrics(y_te, proba_w)))

    print("\n== Results ==")
    print(pretty_metrics_table(rows))
    print("weights =", np.round(w, 4))
    print(f"Total time: {time.perf_counter() - t0_all:.2f}s")

    
    os.makedirs(os.path.join(ROOT, "outputs", "metrics"), exist_ok=True)
    out_csv = os.path.join(ROOT, "outputs", "metrics", f"pmlb_{name}.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("method,acc,f1,auc\n")
        for mname, m in rows:
            acc = m.get("acc", float("nan"))
            f1  = m.get("f1", float("nan"))
            auc = m.get("auc", float("nan"))
            f.write(f"{mname},{acc},{f1},{auc}\n")
    print(f"[saved] {out_csv}")

if __name__ == "__main__":
    for ds in DATASETS:
        run_on_dataset(ds)
