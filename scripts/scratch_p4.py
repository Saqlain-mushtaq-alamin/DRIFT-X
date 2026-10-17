import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.random_streams import generate_random_drift_stream
from driftx.experiment.load_config import load_frozen
from driftx.experiment.runner import ExperimentRunner
from scripts.run_budget_curves import Z_SHAP

print("Z_SHAP:", Z_SHAP)

for s in [1, 2]:
    df, cps, info = generate_random_drift_stream(s, drift_mode="mixed", return_info=True)
    results = []
    for thresh in Z_SHAP:
        cfg = load_frozen({
            "project": {"seed": s, "n_seeds": 1},
            "policy": {"active_policies": ["p4_shap_rank"], "rank_change_threshold": thresh},
            "mlflow": {"enabled": False},
            "output": {"save_csv": False, "export_artifacts": False},
            "model": {"params": {"n_jobs": 1}},
        })
        ev = ExperimentRunner(cfg).run_all(df_override=df)
        ev_active = ev[ev["window_id"] > 0]
        n_retrains = int(ev_active["retrained"].sum())
        results.append((thresh, n_retrains))
    print(f"Seed {s}:", results)
