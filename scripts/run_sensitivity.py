"""
Sensitivity analysis for λ (threshold sensitivity) and k (lookback).

Tests how robust P5 is to different hyperparameter choices.
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


def run_sensitivity(config_path: Optional[str] = None):
    if config_path is None:
        cfg_file = PROJECT_ROOT / "configs" / "default.yaml"
    else:
        cfg_file = Path(config_path)
    
    with open(cfg_file, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    
    all_results = []
    
    for lambda_val, lookback in product(LAMBDA_VALUES, LOOKBACK_VALUES):
        logger.info(f"Testing sensitivity hyperparameter grid point: λ={lambda_val}, k={lookback}")
        
        config = copy.deepcopy(base_config)
        config["fusion"]["adaptive_lambda"] = lambda_val
        config["fusion"]["adaptive_lookback"] = lookback
        config["policy"]["active_policies"] = ["p5_dis_fused"]
        config["project"]["n_seeds"] = min(2, config.get("project", {}).get("n_seeds", 2))  # Speed up grid search
        config.setdefault("mlflow", {})["experiment_name"] = "driftx_sensitivity"
        
        if mlflow is not None:
            try:
                mlflow.set_tracking_uri(config["mlflow"].get("tracking_uri", "mlruns"))
                mlflow.set_experiment(config["mlflow"]["experiment_name"])
            except Exception as e:
                logger.warning(f"MLflow setup warning: {e}")
        
        runner = ExperimentRunner(config)
        # Use a dedicated filename so we don't overwrite experiment_results.csv
        results = runner.run_all(output_filename="sensitivity_run_temp.csv")
        results["lambda"] = lambda_val
        results["lookback"] = lookback
        
        all_results.append(results)
    
    combined = pd.concat(all_results, ignore_index=True)
    out_dir = Path(base_config.get("output", {}).get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sensitivity_results.csv"
    combined.to_csv(out_file, index=False)
    logger.info(f"Saved sensitivity results to {out_file}")
    
    # Print grid summary
    summary = combined.groupby(["lambda", "lookback"]).agg({
        "accuracy": "mean",
        "cumulative_cost": "last",
        "cumulative_retrains": "last",
    }).reset_index()
    logger.info("\n=== Sensitivity Grid Summary ===")
    logger.info("\n" + str(summary.head(10)))
    return combined


if __name__ == "__main__":
    run_sensitivity()
