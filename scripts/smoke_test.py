"""Quick smoke test with synthetic drifting data for Phase 6 ExperimentRunner."""
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmokeTest")


def run_smoke_test():
    logger.info("=== Starting Phase 6 ExperimentRunner Smoke Test ===")
    
    # 1. Generate synthetic dataset with simulated drift
    np.random.seed(42)
    n = 4000
    timestamps = np.arange(n)
    X = np.random.randn(n, 6)
    # Introduce feature drift in second half
    X[2000:, :3] += 1.5
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y[2000:] = (X[2000:, 2] + X[2000:, 3] > 1.0).astype(int)
    
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(6)])
    df["timestamp"] = timestamps
    df["target"] = y
    
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    test_csv = raw_dir / "synthetic_test.csv"
    df.to_csv(test_csv, index=False)
    logger.info(f"Synthetic test dataset saved to {test_csv} (shape: {df.shape})")
    
    # 2. Minimal test config
    config = {
        "project": {"name": "driftx_smoke", "seed": 42, "n_seeds": 1},
        "data": {
            "dataset": "synthetic_test",
            "timestamp_col": "timestamp",
            "target_col": "target",
            "window_size": 8,
            "min_window_samples": 400,
        },
        "model": {
            "type": "xgboost",
            "params": {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1}
        },
        "detection": {
            "statistical_method": "psi",
            "psi_threshold": 0.15,
            "ks_alpha": 0.05
        },
        "explainability": {
            "shap_method": "tree",
            "max_shap_samples": 250,
            "magnitude_threshold": 0.08,
            "rank_change_threshold": 0.25
        },
        "fusion": {
            "alpha": 0.3,
            "beta": 0.3,
            "gamma": 0.4,
            "adaptive_lookback": 3,
            "adaptive_lambda": 1.5
        },
        "policy": {
            "fixed_schedule_interval": 2,
            "active_policies": ["p0_never", "p1_fixed", "p5_dis_fused"]
        },
        "mlflow": {
            "tracking_uri": str(PROJECT_ROOT / "mlruns"),
            "experiment_name": "smoke_test_experiment"
        },
        "output": {
            "results_dir": str(PROJECT_ROOT / "results")
        }
    }
    
    runner = ExperimentRunner(config)
    results = runner.run_all(df_override=df)
    
    logger.info(f"\nSmoke test complete! Logged {len(results)} rows.")
    assert len(results) > 0, "No results produced by ExperimentRunner!"
    assert "accuracy" in results.columns, "Accuracy column missing in results!"
    assert "policy" in results.columns, "Policy column missing in results!"
    
    logger.info("✓ Smoke Test PASSED Successfully!")


if __name__ == "__main__":
    run_smoke_test()
