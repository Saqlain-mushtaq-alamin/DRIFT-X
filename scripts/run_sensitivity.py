"""
Sensitivity analysis for λ (threshold sensitivity) and k (lookback).

Tests how robust P5 is to different hyperparameter choices.

IMPORTANT — Honest protocol:
  This script evaluates the sensitivity grid on a SEPARATE holdout stream
  (seed=999, distinct from the evaluation seeds 42–46) so that the "sweet
  spot" is NOT found by looking at the evaluation stream.  The evaluation
  results in experiment_results.csv must still use the parameters in
  default.yaml; this script only provides a diagnostic of parameter
  stability, NOT a licence to change those parameters.

  If the best grid point from this script differs from default.yaml, that
  discrepancy should be disclosed in the paper.
"""
import sys
import yaml
import copy
import logging
from pathlib import Path
from itertools import product
from typing import Optional
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mlflow
except ImportError:
    mlflow = None

from driftx.experiment.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SensitivityAnalysis")

LAMBDA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
LOOKBACK_VALUES = [2, 3, 5, 7]  # 10 removed — requires >10 windows to exit warmup

# Holdout stream seed — MUST NOT overlap with evaluation seeds 42–46.
HOLDOUT_SEED = 999
EVAL_SEEDS = set(range(42, 47))


def _load_holdout_stream(seed: int = HOLDOUT_SEED, n_windows: int = 20) -> pd.DataFrame:
    """Generate a synthetic holdout stream on a seed not used in evaluation."""
    from scripts.download_data import generate_synthetic_drift_data
    out_path = PROJECT_ROOT / "data" / "raw" / f"sensitivity_holdout_seed{seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_drift_data(
        output_path=str(out_path),
        n_windows=n_windows,
        samples_per_window=1500,
        n_features=15,
        seed=seed,
        drift_intensity=0.4,
    )
    logger.info(f"Loaded sensitivity holdout stream: {len(df):,} rows, seed={seed}")
    return df


def run_sensitivity(config_path: Optional[str] = None):
    if config_path is None:
        cfg_file = PROJECT_ROOT / "configs" / "default.yaml"
    else:
        cfg_file = Path(config_path)
    
    with open(cfg_file, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    if HOLDOUT_SEED in EVAL_SEEDS:
        raise ValueError(
            f"HOLDOUT_SEED={HOLDOUT_SEED} overlaps with evaluation seeds {sorted(EVAL_SEEDS)}. "
            "Choose a different holdout seed."
        )
    
    # Load held-out stream — NEVER the evaluation stream.
    logger.info(
        f"Loading sensitivity holdout stream (seed={HOLDOUT_SEED}). "
        f"Evaluation seeds {sorted(EVAL_SEEDS)} are kept separate."
    )
    df_holdout = _load_holdout_stream(seed=HOLDOUT_SEED, n_windows=20)
    
    all_results = []
    
    # Use n_seeds=3 for more stable estimates at each grid point.
    GRID_SEEDS = 3
    
    for lambda_val, lookback in product(LAMBDA_VALUES, LOOKBACK_VALUES):
        logger.info(f"Testing sensitivity hyperparameter grid point: λ={lambda_val}, k={lookback}")
        
        config = copy.deepcopy(base_config)
        config["fusion"]["adaptive_lambda"] = lambda_val
        config["fusion"]["adaptive_lookback"] = lookback
        config["policy"]["active_policies"] = ["p5_dis_fused"]
        # Use holdout-only seeds (not overlapping 42-46)
        config["project"]["seed"] = HOLDOUT_SEED
        config["project"]["n_seeds"] = GRID_SEEDS
        config.setdefault("mlflow", {})["experiment_name"] = "driftx_sensitivity"
        
        if mlflow is not None:
            try:
                mlflow.set_tracking_uri(config["mlflow"].get("tracking_uri", "mlruns"))
                mlflow.set_experiment(config["mlflow"]["experiment_name"])
            except Exception as e:
                logger.warning(f"MLflow setup warning: {e}")
        
        runner = ExperimentRunner(config)
        # Use a dedicated filename so we don't overwrite experiment_results.csv
        results = runner.run_all(
            df_override=df_holdout,
            output_filename="sensitivity_run_temp.csv",
        )
        results["lambda"] = lambda_val
        results["lookback"] = lookback
        
        all_results.append(results)
    
    combined = pd.concat(all_results, ignore_index=True)
    out_dir = Path(base_config.get("output", {}).get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sensitivity_results.csv"
    combined.to_csv(out_file, index=False)
    logger.info(f"Saved sensitivity results to {out_file}")
    
    # Print grid summary — excludes window_id==0 (NaN accuracy by design)
    eval_rows = combined[combined["window_id"] > 0]
    summary = eval_rows.groupby(["lambda", "lookback"]).agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_retrains=("retrain_count", "last"),
    ).reset_index().round(4)

    logger.info("\n=== Sensitivity Grid Summary (holdout stream, seed=%d) ===", HOLDOUT_SEED)
    logger.info(f"\n{summary.to_string(index=False)}")

    # Check against default config and warn if best differs
    default_lambda = base_config.get("fusion", {}).get("adaptive_lambda", 1.5)
    default_k = base_config.get("fusion", {}).get("adaptive_lookback", 3)
    best_row = summary.sort_values("mean_accuracy", ascending=False).iloc[0]
    if best_row["lambda"] != default_lambda or best_row["lookback"] != default_k:
        logger.warning(
            f"\n[DISCLOSURE REQUIRED] Best grid point on holdout: "
            f"λ={best_row['lambda']}, k={int(best_row['lookback'])}, "
            f"acc={best_row['mean_accuracy']:.4f}. "
            f"Default config uses λ={default_lambda}, k={default_k}. "
            "If you change the config to use the holdout-optimal values, "
            "you MUST re-run all evaluation scripts before reporting results."
        )
    else:
        logger.info(
            f"\nDefault config (λ={default_lambda}, k={default_k}) matches the holdout-optimal "
            "grid point — no disclosure required."
        )
    return combined


if __name__ == "__main__":
    run_sensitivity()

