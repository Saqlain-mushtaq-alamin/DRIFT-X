"""
Ablation study: Isolate which DIS component matters most.

Variants:
1. Full DIS (alpha=0.4, beta=0.4, gamma=0.2) -- matches default.yaml after Bug-3 fix
2. StatOnly (alpha=1.0, beta=0.0, gamma=0.0) -- only statistical drift
3. MagOnly (alpha=0.0, beta=1.0, gamma=0.0) -- only SHAP magnitude
4. RankOnly (alpha=0.0, beta=0.0, gamma=1.0) -- only SHAP rank-change
5. Stat+Mag (alpha=0.5, beta=0.5, gamma=0.0) -- no rank-change
6. Stat+Rank (alpha=0.5, beta=0.0, gamma=0.5) -- no magnitude
7. Mag+Rank (alpha=0.0, beta=0.5, gamma=0.5) -- no statistical
"""
import sys
import yaml
import copy
import logging
from pathlib import Path
from typing import Optional, Dict, Any
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
logger = logging.getLogger("AblationStudy")

ABLATION_CONFIGS = {
    "full_dis":     {"alpha": 0.4, "beta": 0.4, "gamma": 0.2, "gating_mode": "two_channel"},
    "stat_only":    {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "gating_mode": "none"},
    "mag_only":     {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "gating_mode": "none"},
    "rank_only":    {"alpha": 0.0, "beta": 0.0, "gamma": 1.0, "gating_mode": "none"},
    "stat_mag":     {"alpha": 0.5, "beta": 0.5, "gamma": 0.0, "gating_mode": "two_channel"},
    "stat_rank":    {"alpha": 0.5, "beta": 0.0, "gamma": 0.5, "gating_mode": "two_channel"},
    "mag_rank":     {"alpha": 0.0, "beta": 0.5, "gamma": 0.5, "gating_mode": "two_channel"},
}


def run_ablation(config_path: Optional[str] = None):
    if config_path is None:
        cfg_file = PROJECT_ROOT / "configs" / "default.yaml"
    else:
        cfg_file = Path(config_path)
    
    with open(cfg_file, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    
    all_results = []
    
    for ablation_name, weights in ABLATION_CONFIGS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Ablation: {ablation_name} | Weights: {weights}")
        logger.info(f"{'='*50}")
        
        config = copy.deepcopy(base_config)
        config["fusion"].update(weights)
        
        # Only run P5 (DIS-Fused) for ablation
        config["policy"]["active_policies"] = ["p5_dis_fused"]
        config.setdefault("mlflow", {})["experiment_name"] = "driftx_ablation"
        
        if mlflow is not None:
            try:
                mlflow.set_tracking_uri(config["mlflow"].get("tracking_uri", "mlruns"))
                mlflow.set_experiment(config["mlflow"]["experiment_name"])
            except Exception as e:
                logger.warning(f"MLflow setup warning: {e}")
        
        runner = ExperimentRunner(config)
        # Use a dedicated filename so we don't overwrite experiment_results.csv
        results = runner.run_all(output_filename="ablation_run_temp.csv")
        results["ablation"] = ablation_name
        results["alpha"] = weights["alpha"]
        results["beta"] = weights["beta"]
        results["gamma"] = weights["gamma"]
        
        all_results.append(results)
    
    # Combine and save
    combined = pd.concat(all_results, ignore_index=True)
    out_dir = Path(base_config.get("output", {}).get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "ablation_results.csv"
    combined.to_csv(output_path, index=False)
    logger.info(f"\nAblation results saved to {output_path}")
    
    # Print summary table
    summary = combined.groupby("ablation").agg({
        "accuracy": ["mean", "std"],
        "f1": ["mean", "std"],
        "cumulative_cost": "last",
        "cumulative_retrains": "last",
    })
    logger.info("\n=== Ablation Summary ===")
    logger.info("\n" + str(summary))
    return combined


if __name__ == "__main__":
    run_ablation()
