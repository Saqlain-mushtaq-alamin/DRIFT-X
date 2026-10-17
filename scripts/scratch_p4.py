import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.random_streams import generate_random_drift_stream
from driftx.experiment.load_config import load_frozen
from driftx.experiment.runner import ExperimentRunner

for s in [1, 2]:
    df, cps, info = generate_random_drift_stream(s, drift_mode="mixed", return_info=True)
    cfg = load_frozen({
        "project": {"seed": s, "n_seeds": 1},
        "policy": {"active_policies": ["p4_shap_rank"], "rank_change_threshold": 999.0},
        "mlflow": {"enabled": False},
        "output": {"save_csv": False, "export_artifacts": False},
        "model": {"params": {"n_jobs": 1}},
    })
    ev = ExperimentRunner(cfg).run_all(df_override=df)
    ev_active = ev[ev["window_id"] > 0]
    print(f"=== SEED {s} cps={cps} ===")
    for idx, row in ev_active.iterrows():
        print(f"w={int(row['window_id'])}: z_rank={row.get('z_rank', 0.0):.3f}, score={row.get('shap_rank_change_score', 0.0):.3f}")
