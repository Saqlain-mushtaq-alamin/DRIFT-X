import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scripts.random_streams import generate_random_drift_stream
from driftx.experiment.load_config import load_frozen
from driftx.experiment.runner import ExperimentRunner
from joblib import Parallel, delayed

def geo(a: float, b: float, n: int):
    return [float(x) for x in np.round(np.geomspace(a, b, n), 3)]

grid_cand = geo(0.1, 6000, 14)
print("Grid:", grid_cand)

def eval_one(s, thresh):
    df, cps, info = generate_random_drift_stream(s, drift_mode="mixed", return_info=True)
    cfg = load_frozen({
        "project": {"seed": s, "n_seeds": 1},
        "policy": {"active_policies": ["p4_shap_rank"], "rank_change_threshold": thresh},
        "mlflow": {"enabled": False},
        "output": {"save_csv": False, "export_artifacts": False},
        "model": {"params": {"n_jobs": 1}},
    })
    ev = ExperimentRunner(cfg).run_all(df_override=df)
    ev_active = ev[ev["window_id"] > 0]
    return s, thresh, int(ev_active["retrained"].sum())

if __name__ == "__main__":
    records = []
    for s in [1, 2]:
        res = Parallel(n_jobs=4)(delayed(eval_one)(s, t) for t in grid_cand)
        for _, t, r in res:
            records.append({"stream": s, "thresh": t, "retrains": r})
    
    df = pd.DataFrame(records)
    span = (df.groupby(["stream"])["retrains"].agg(["min", "max", "nunique"]))
    print("\nPer stream span:")
    print(span)
    print("\nMean span:")
    print(span.mean())
