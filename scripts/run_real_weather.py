"""Run evaluation of P2, P5, P7, wstat and baselines on NOAA Weather Real Stream benchmark."""
import sys
import yaml
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner, get_git_commit, get_git_dirty, get_frozen_hash
from driftx.experiment.load_config import load_frozen
from scripts.run_budget_curves import GRID_P7, GRID_WSTAT, Z_STAT, GRIDS, curve_at

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RealWeatherBenchmark")


def load_weather_config() -> Dict[str, Any]:
    with open(PROJECT_ROOT / "configs" / "datasets" / "weather.yaml") as f:
        w_cfg = yaml.safe_load(f)
    return w_cfg


def run_one_weather(pol: str, sec: str, key: str, val: Any) -> Dict[str, Any]:
    w_cfg = load_weather_config()
    real_pol, extra_fusion = pol, {}
    if pol == "p5_stat_only":
        real_pol = "p5_dis_fused"
        extra_fusion = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "gating_mode": "none",
                        "stat_weighting": "none", "clip_max": 1000.0}
    elif pol == "p5_wstat_only":
        real_pol = "p5_dis_fused"
        extra_fusion = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "gating_mode": "none",
                        "stat_weighting": "shap", "clip_max": 1000.0}

    cfg = load_frozen({
        "project": {"seed": 42, "n_seeds": 1},
        "policy": {"active_policies": [real_pol]},
        "mlflow": {"enabled": False},
        "output": {"save_csv": False, "export_artifacts": False},
        "model": {"params": {"n_jobs": 1}},
    })
    cfg["data"].update(w_cfg["data"])
    if extra_fusion:
        cfg["fusion"].update(extra_fusion)
    if key != "unused":
        cfg[sec][key] = val
        if sec == "fusion" and key == "dis_threshold":
            cfg["fusion"]["fixed_threshold"] = val

    ev = ExperimentRunner(cfg).run_all()
    ev_active = ev[ev["window_id"] > 0]

    return {
        "dataset": "weather", "policy": pol, "param_key": key, "param_val": val,
        "retrains": int(ev_active["retrained"].sum()),
        "acc": float(ev_active["accuracy"].mean()),
        "n_eval": len(ev_active),
    }


def main():
    logger.info("Running Real Stream Benchmark on NOAA Weather (NEweather)...")
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    # 1. Standard operating points
    standard_policies = [
        ("p0_never", "policy", "unused", 0),
        ("p1_fixed", "policy", "fixed_schedule_interval", 3),
        ("p2_drift_only", "policy", "unused", 0),
        ("p5_dis_fused", "fusion", "unused", 0),
        ("p7_weighted_ks", "policy", "unused", 0),
        ("p5_wstat_only", "fusion", "dis_threshold", 1.0),
    ]
    logger.info("Evaluating standard default-threshold operating points...")
    std_rows = [run_one_weather(*j) for j in standard_policies]
    df_std = pd.DataFrame(std_rows)
    logger.info("\nStandard Operating Points on Weather:\n" + df_std.to_string(index=False))
    df_std.to_csv(out_dir / "real_weather_standard.csv", index=False)

    # 2. Dense budget curves for P2, P5, P7, wstat
    sweep_jobs = [
        ("p0_never", "policy", "unused", 0),
        ("p1_fixed", "policy", "fixed_schedule_interval", 1),
    ]
    # Add sweeps
    for v in Z_STAT:
        sweep_jobs.append(("p2_drift_only", "policy", "ks_threshold", v))
    for v in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.9, 5.0]:
        sweep_jobs.append(("p5_dis_fused", "fusion", "dis_threshold", v))
    for v in GRID_P7:
        sweep_jobs.append(("p7_weighted_ks", "policy", "weighted_ks_threshold", v))
    for v in GRID_WSTAT:
        sweep_jobs.append(("p5_wstat_only", "fusion", "dis_threshold", v))

    logger.info(f"Running budget curve sweeps ({len(sweep_jobs)} configurations) on Weather...")
    sweep_results = Parallel(n_jobs=4, verbose=5)(delayed(run_one_weather)(*j) for j in sweep_jobs)
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(out_dir / "real_weather_budget_curves_raw.csv", index=False)

    # Compute matched-budget metrics
    p0_acc = df_sweep[df_sweep.policy == "p0_never"]["acc"].iloc[0]
    always_acc = df_sweep[(df_sweep.policy == "p1_fixed") & (df_sweep.param_val == 1)]["acc"].iloc[0]
    n_eval = int(df_sweep["n_eval"].iloc[0])

    curve_rows = []
    for pol, g in df_sweep.groupby("policy"):
        if pol == "p0_never":
            continue
        at = {f"acc@{B}": curve_at(g, p0_acc, always_acc, n_eval, B) for B in (3, 4, 5)}
        for B in (3, 4, 5):
            at[f"near@{B}"] = bool((g["retrains"] - B).abs().min() <= 1)
        budgets = range(1, 11)
        auc = float(np.mean([curve_at(g, p0_acc, always_acc, n_eval, B) for B in budgets]))
        auc_cov = float(np.mean([(g["retrains"] - B).abs().min() <= 1 for B in budgets]))
        curve_rows.append({
            "policy": pol, **at, "auc": auc, "auc_cov": auc_cov,
            "min_retrains": g["retrains"].min(), "max_retrains": g["retrains"].max(),
        })

    df_curves = pd.DataFrame(curve_rows).sort_values("auc", ascending=False)
    df_curves.to_csv(out_dir / "real_weather_budget_curves_summary.csv", index=False)
    logger.info("\nMatched-Budget Curves on Real Weather Stream:\n" + df_curves.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
