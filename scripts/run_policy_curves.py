"""
Policy operating-point curves for DRIFT-X.

Problem (Issue 3)
-----------------
Each policy has at least one tunable parameter that controls the
accuracy-vs-retrain tradeoff:

  P1 (fixed schedule) : fixed_schedule_interval  in {1,2,3,4,5,6,8,10}
  P2 (KS drift only)  : ks_alpha                 in {0.01,0.02,0.05,0.10,0.20}
  P3 (SHAP magnitude) : magnitude_threshold       in {0.01,0.02,0.03,0.05,0.08,0.15}
  P4 (SHAP rank)      : rank_change_threshold     in {0.01,0.02,0.035,0.07,0.12,0.20}
  P5 (DIS-fused)      : adaptive_lambda           in {0.5,1.0,1.5,2.0,2.5,3.0}

Comparing a single operating point (the default value) cannot demonstrate
dominance.  This script generates a curve of (mean_retrains, mean_accuracy)
for each policy over its parameter sweep, producing data for a Pareto-front
comparison plot.  A policy is only dominant if its curve consistently lies
above the other curves at the same retrain count.

Usage
-----
  python scripts/run_policy_curves.py [--seeds 3] [--config configs/default.yaml]

Output
------
  results/policy_curves.csv         -- raw per-seed operating points
  results/policy_curves_summary.csv -- mean accuracy +/- std per operating point
"""
import sys
import copy
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PolicyCurves")

# ---------------------------------------------------------------------------
# Parameter sweeps per policy
# ---------------------------------------------------------------------------
POLICY_SWEEPS: Dict[str, Dict[str, Any]] = {
    "p1_fixed": {
        "param_name": "fixed_schedule_interval",
        "param_values": [1, 2, 3, 4, 6, 8, 10],
        "config_keys": ("policy", "fixed_schedule_interval"),
    },
    "p2_drift_only": {
        "param_name": "ks_threshold",
        "param_values": [1.5, 2.0, 2.5, 3.0, 4.0],
        "config_keys": ("detection", "ks_threshold"),
    },
    "p5_dis_fused": {
        "param_name": "fixed_threshold",
        "param_values": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "config_keys": ("fusion", "fixed_threshold"),
    },
    "p6_performance_drop": {
        "param_name": "delta",
        "param_values": [0.01, 0.02, 0.03, 0.05, 0.08],
        "config_keys": ("policy", "p6_delta"),
    },
    "p8_random_budget": {
        "param_name": "retrain_prob",
        "param_values": [0.05, 0.1, 0.2, 0.35, 0.5],
        "config_keys": ("policy", "p8_retrain_prob"),
    },
}


def _set_nested(d: dict, keys: tuple, value) -> None:
    """Set d[keys[0]][keys[1]] = value (two-level only)."""
    d[keys[0]][keys[1]] = value


def run_policy_curves(
    config_path: str = None,
    n_seeds: int = 3,
) -> pd.DataFrame:
    """Sweep each policy's tunable parameter and record (retrains, accuracy) pairs."""
    if config_path is None:
        cfg_file = PROJECT_ROOT / "configs" / "default.yaml"
    else:
        cfg_file = Path(config_path)

    with open(cfg_file, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    all_rows = []

    for policy_id, sweep in POLICY_SWEEPS.items():
        param_name = sweep["param_name"]
        param_values = sweep["param_values"]
        cfg_keys = sweep["config_keys"]

        logger.info("\n%s", "=" * 60)
        logger.info("Sweeping %s over %s: %s", policy_id, param_name, param_values)

        for pval in param_values:
            config = copy.deepcopy(base_config)
            config["policy"]["active_policies"] = [policy_id]
            config["project"]["n_seeds"] = n_seeds
            _set_nested(config, cfg_keys, pval)
            config.setdefault("mlflow", {})["experiment_name"] = "driftx_curves"

            runner = ExperimentRunner(config)
            results = runner.run_all(
                output_filename=f"curves_temp_{policy_id}.csv"
            )

            # Exclude window 0 (accuracy=NaN by design) before aggregating
            eval_rows = results[results["window_id"] > 0]
            per_seed = eval_rows.groupby("seed").agg(
                mean_acc=("accuracy", "mean"),
                total_retrains=("retrained", "sum"),
            ).reset_index()

            for _, sr in per_seed.iterrows():
                all_rows.append({
                    "policy": policy_id,
                    "param_name": param_name,
                    "param_value": pval,
                    "seed": int(sr["seed"]),
                    "mean_accuracy": round(float(sr["mean_acc"]), 6),
                    "total_retrains": int(sr["total_retrains"]),
                })
            logger.info(
                "  %s @ %s=%s: acc=%.4f +/- %.4f, retrains=%.1f",
                policy_id, param_name, pval,
                per_seed["mean_acc"].mean(),
                per_seed["mean_acc"].std() if len(per_seed) > 1 else 0.0,
                per_seed["total_retrains"].mean(),
            )

    df = pd.DataFrame(all_rows)
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / "policy_curves.csv"
    df.to_csv(raw_path, index=False)

    # Summary: mean +/- std per (policy, param_value)
    summary = (
        df.groupby(["policy", "param_name", "param_value"])
        .agg(
            mean_accuracy=("mean_accuracy", "mean"),
            std_accuracy=("mean_accuracy", "std"),
            mean_retrains=("total_retrains", "mean"),
            std_retrains=("total_retrains", "std"),
        )
        .reset_index()
        .round(4)
    )
    summary_path = out_dir / "policy_curves_summary.csv"
    summary.to_csv(summary_path, index=False)

    logger.info("Policy curves saved to %s", raw_path)
    logger.info("Summary saved to %s", summary_path)

    print("\n=== Policy Operating-Point Curves ===")
    print(summary.to_string(index=False))
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 6))
        for pol, grp in summary.groupby("policy"):
            grp_sorted = grp.sort_values("mean_retrains")
            plt.plot(grp_sorted["mean_retrains"], grp_sorted["mean_accuracy"], marker="o", label=pol)
        plt.xlabel("Net Retrain Count")
        plt.ylabel("Prequential Out-of-Sample Accuracy")
        plt.title("DRIFT-X Policy Operating-Point Tradeoff Curves")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plot_path = out_dir / "policy_curves.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info("Saved Pareto curves plot to %s", plot_path)
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="DRIFT-X Policy Accuracy-vs-Retrains Operating-Point Curves"
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of seeds per operating point (default: 3)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config YAML path (default: configs/default.yaml)"
    )
    args = parser.parse_args()
    run_policy_curves(config_path=args.config, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
