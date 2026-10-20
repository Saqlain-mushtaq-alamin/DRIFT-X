"""Tune P5 fusion hyperparameters on tuning streams (seeds 1-10) using budget AUC.

Evaluates candidate gating parameters and threshold grids, computes the
accuracy-retrains curve AUC anchored at P0, and freezes the optimal parameters into configs/frozen.yaml.
"""
import copy
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner, compute_config_hash, get_git_commit
from driftx.experiment.load_config import load_frozen
from scripts.random_streams import generate_random_drift_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TuneHoldout")


def run_single_eval(s, pol, gating_mode, gating_thresh, dis_thresh, p2_thresh=20.0):
    df, _ = generate_random_drift_stream(seed=s, drift_mode="mixed")
    cfg = load_frozen({
        "project": {"seed": s, "n_seeds": 1},
        "policy": {"active_policies": [pol]},
        "mlflow": {"enabled": False},
        "output": {"save_csv": False, "export_artifacts": False},
        "model": {"params": {"n_jobs": 1}},
    })
    if pol == "p5_dis_fused":
        cfg["fusion"]["gating_mode"] = gating_mode
        cfg["fusion"]["gating_channel_threshold"] = gating_thresh
        cfg["fusion"]["dis_threshold"] = dis_thresh
        cfg["fusion"]["fixed_threshold"] = dis_thresh
    elif pol == "p2_drift_only":
        cfg["policy"]["ks_threshold"] = p2_thresh
        cfg["policy"]["fixed_threshold"] = p2_thresh

    runner = ExperimentRunner(cfg)
    res = runner.run_all(df_override=df, output_filename=f"tune_temp_{s}.csv")
    eval_rows = res[res["window_id"] > 0]
    retrains = int(eval_rows["retrained"].sum())
    acc = float(eval_rows["accuracy"].mean())
    return {
        "stream": s,
        "policy": pol,
        "gating_mode": gating_mode,
        "gating_thresh": gating_thresh,
        "dis_thresh": dis_thresh,
        "retrains": retrains,
        "acc": acc,
    }


def curve_at(df_sp, p0_acc, B):
    c = df_sp.groupby("retrains").acc.mean()
    c.loc[0] = p0_acc
    c = c.sort_index()
    # Interpolate along curve, don't extrapolate beyond range
    return float(np.interp(B, c.index, c.values, left=np.nan, right=np.nan))


def main():
    tuning_seeds = range(1, 11)
    logger.info("Evaluating P0 baseline on tuning seeds 1-10...")
    # First get P0 for each seed
    p0_jobs = [delayed(run_single_eval)(s, "p0_never", "none", 0.0, 0.0) for s in tuning_seeds]
    p0_results = Parallel(n_jobs=4)(p0_jobs)
    p0_df = pd.DataFrame(p0_results)
    p0_map = dict(zip(p0_df["stream"], p0_df["acc"]))
    logger.info(f"P0 mean accuracy on tuning seeds: {np.mean(list(p0_map.values())):.4f}")

    # Gating configurations to test for P5
    gating_candidates = [
        ("two_channel", 1.5),
        ("two_channel", 2.0),
        ("two_channel", 2.5),
        ("soft", 1.5),
        ("soft", 2.0),
    ]
    dis_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    all_jobs = []
    for g_mode, g_th in gating_candidates:
        for d_th in dis_thresholds:
            for s in tuning_seeds:
                all_jobs.append(delayed(run_single_eval)(s, "p5_dis_fused", g_mode, g_th, d_th))

    logger.info(f"Running P5 grid search ({len(all_jobs)} runs across 4 workers)...")
    results = Parallel(n_jobs=4)(all_jobs)
    df_res = pd.DataFrame(results)

    # Compute AUC for each gating configuration
    gating_scores = []
    for (g_mode, g_th), group in df_res.groupby(["gating_mode", "gating_thresh"]):
        seed_aucs = []
        seed_acc_at_4 = []
        for s in tuning_seeds:
            s_data = group[group["stream"] == s]
            p0_acc = p0_map[s]
            grid_vals = [curve_at(s_data, p0_acc, b) for b in range(1, 11)]
            # If some points can be interpolated, average available points in budget range 1-10
            valid_vals = [v for v in grid_vals if not np.isnan(v)]
            auc = np.mean(valid_vals) if len(valid_vals) >= 5 else np.nan
            seed_aucs.append(auc)
            acc_4 = curve_at(s_data, p0_acc, 4)
            seed_acc_at_4.append(acc_4)

        mean_auc = np.nanmean(seed_aucs)
        mean_acc4 = np.nanmean(seed_acc_at_4)
        logger.info(f"Gating {g_mode} (thresh={g_th}): Mean Curve AUC={mean_auc:.4f}, acc@4={mean_acc4:.4f}")
        gating_scores.append({
            "gating_mode": g_mode,
            "gating_channel_threshold": g_th,
            "mean_auc": mean_auc,
            "acc_at_4": mean_acc4,
        })

    df_gating = pd.DataFrame(gating_scores).sort_values("mean_auc", ascending=False)
    best_gating = df_gating.iloc[0]
    logger.info(f"Best Gating Configuration: {best_gating['gating_mode']} with thresh={best_gating['gating_channel_threshold']} (AUC={best_gating['mean_auc']:.4f})")

    # Find the best operating threshold for budget 3-5
    best_group = df_res[
        (df_res["gating_mode"] == best_gating["gating_mode"]) &
        (df_res["gating_thresh"] == best_gating["gating_channel_threshold"])
    ]
    summary_by_thresh = best_group.groupby("dis_thresh").agg(
        mean_retrains=("retrains", "mean"),
        mean_acc=("acc", "mean")
    ).reset_index()
    logger.info("\nOperating point by threshold under best gating:")
    logger.info(summary_by_thresh.to_string(index=False))

    # Pick threshold with mean retrains closest to 4 (matched budget target 3-5)
    summary_by_thresh["dist_to_4"] = (summary_by_thresh["mean_retrains"] - 4.0).abs()
    best_point = summary_by_thresh.sort_values(["dist_to_4", "mean_acc"], ascending=[True, False]).iloc[0]
    best_thresh = float(best_point["dis_thresh"])
    logger.info(f"Selected Frozen Operating Threshold: {best_thresh} (Mean retrains: {best_point['mean_retrains']:.1f}, Mean acc: {best_point['mean_acc']:.4f})")

    # Update frozen.yaml
    frozen_path = PROJECT_ROOT / "configs" / "frozen.yaml"
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen_cfg = yaml.safe_load(f)

    frozen_cfg["fusion"]["gating_mode"] = str(best_gating["gating_mode"])
    frozen_cfg["fusion"]["gating_channel_threshold"] = float(best_gating["gating_channel_threshold"])
    frozen_cfg["fusion"]["dis_threshold"] = best_thresh
    frozen_cfg["fusion"]["fixed_threshold"] = best_thresh

    with open(frozen_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(frozen_cfg, f, sort_keys=False)

    commit_sha = get_git_commit()
    config_hash = compute_config_hash(frozen_cfg)
    logger.info(f"Updated {frozen_path} successfully!")
    logger.info(f"Frozen Git Commit: {commit_sha}")
    logger.info(f"Frozen Config Hash: {config_hash}")


if __name__ == "__main__":
    main()
