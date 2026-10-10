"""Tune P2 and P5 hyperparameters on independent tuning streams (seeds 1-10) only.

Objective: accuracy - 0.01 * net_retrains (cost-aware objective).
The best parameters are written to configs/frozen.yaml and permanently frozen.
"""
import copy
import hashlib
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from driftx.experiment.runner import ExperimentRunner
from scripts.random_streams import generate_random_drift_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TuneHoldout")


def evaluate_config_on_tuning_streams(config: dict, tuning_seeds=range(1, 11)) -> dict:
    """Evaluate a policy configuration across tuning streams."""
    results = []
    for s in tuning_seeds:
        df, cps = generate_random_drift_stream(seed=s)
        cfg = copy.deepcopy(config)
        cfg["project"]["seed"] = s
        cfg["project"]["n_seeds"] = 1
        runner = ExperimentRunner(cfg)
        res = runner.run_all(df_override=df, output_filename=f"tune_temp_{s}.csv")
        eval_rows = res[res["window_id"] > 0]
        acc = eval_rows["accuracy"].mean()
        retrains = eval_rows["retrained"].sum()
        results.append({"stream": s, "accuracy": acc, "retrains": retrains})
    
    df_res = pd.DataFrame(results)
    mean_acc = df_res["accuracy"].mean()
    mean_ret = df_res["retrains"].mean()
    score = mean_acc - 0.01 * mean_ret
    return {
        "mean_accuracy": mean_acc,
        "mean_retrains": mean_ret,
        "score": score,
        "per_stream": df_res,
    }


def main():
    base_cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # 1. Tune P2 ks_threshold
    logger.info("--- Tuning P2 ks_threshold ---")
    p2_candidates = [3.0, 5.0, 10.0, 20.0, 50.0]
    best_p2_thresh = 3.0
    best_p2_score = -float("inf")
    p2_results = {}

    for thresh in p2_candidates:
        cfg = copy.deepcopy(base_config)
        cfg["policy"]["active_policies"] = ["p2_drift_only"]
        cfg["policy"]["ks_threshold"] = thresh
        cfg["policy"]["fixed_threshold"] = thresh
        res = evaluate_config_on_tuning_streams(cfg)
        logger.info(f"P2 threshold={thresh}: acc={res['mean_accuracy']:.4f}, retrains={res['mean_retrains']:.1f}, score={res['score']:.4f}")
        p2_results[thresh] = res
        if res["score"] > best_p2_score:
            best_p2_score = res["score"]
            best_p2_thresh = thresh

    logger.info(f"Best P2 ks_threshold: {best_p2_thresh} (score={best_p2_score:.4f})")

    # 2. Tune P5 gating & thresholds
    logger.info("\n--- Tuning P5 Fusion Parameters ---")
    p5_grid = [
        {"gating_mode": "two_channel", "gating_channel_threshold": 1.5, "fixed_threshold": 2.5},
        {"gating_mode": "two_channel", "gating_channel_threshold": 2.0, "fixed_threshold": 3.0},
        {"gating_mode": "soft", "gating_channel_threshold": 1.5, "fixed_threshold": 2.5},
        {"gating_mode": "soft", "gating_channel_threshold": 2.0, "fixed_threshold": 3.0},
        {"gating_mode": "soft", "gating_channel_threshold": 2.0, "fixed_threshold": 3.5},
    ]
    best_p5_cfg = p5_grid[0]
    best_p5_score = -float("inf")
    p5_results = []

    for cand in p5_grid:
        cfg = copy.deepcopy(base_config)
        cfg["policy"]["active_policies"] = ["p5_dis_fused"]
        cfg["fusion"].update(cand)
        res = evaluate_config_on_tuning_streams(cfg)
        logger.info(f"P5 {cand}: acc={res['mean_accuracy']:.4f}, retrains={res['mean_retrains']:.1f}, score={res['score']:.4f}")
        cand_record = {**cand, **res}
        p5_results.append(cand_record)
        if res["score"] > best_p5_score:
            best_p5_score = res["score"]
            best_p5_cfg = cand

    logger.info(f"Best P5 config: {best_p5_cfg} (score={best_p5_score:.4f})")

    # 3. Create frozen.yaml with selected parameters
    frozen_config = copy.deepcopy(base_config)
    frozen_config["policy"]["ks_threshold"] = best_p2_thresh
    frozen_config["policy"]["fixed_threshold"] = best_p2_thresh
    frozen_config["fusion"].update(best_p5_cfg)
    
    frozen_path = PROJECT_ROOT / "configs" / "frozen.yaml"
    with open(frozen_path, "w", encoding="utf-8") as f:
        yaml.dump(frozen_config, f, default_flow_style=False)

    # Compute frozen hash
    hasher = hashlib.sha256()
    with open(frozen_path, "rb") as f:
        hasher.update(f.read())
    frozen_hash = hasher.hexdigest()[:12]
    logger.info(f"Frozen configuration written to {frozen_path} with SHA256={frozen_hash}")


if __name__ == "__main__":
    main()
