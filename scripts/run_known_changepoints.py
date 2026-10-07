"""Known change-point detection experiment for DRIFT-X.

Purpose
-------
Tests whether the drift detection signals correctly identify *known* concept
change points in a controlled synthetic stream.  Provides two complementary
metrics that null-stream testing alone cannot give:

  1. **Detection delay**: how many windows after the true change point does
     each policy first trigger a retrain?  Lower is better.
  2. **False trigger rate**: fraction of windows WITHOUT a change point where
     a retrain is triggered.  Should be close to 0.

Experimental design
-------------------
The stream has explicit phase boundaries:
  - Windows 0..4:   Phase A  (stable, logit = X0·0.8 − X1·0.5)
  - Window  5:      CHANGE POINT 1  (abrupt concept + covariate shift)
  - Windows 5..9:   Phase B  (drifted, logit = −X0·0.6 + X2·0.7)
  - Window  10:     CHANGE POINT 2  (second abrupt shift)
  - Windows 10..N:  Phase C  (further drift, logit = X1·0.5 + X3·0.4)

Ground truth labels are attached to each result row so downstream analysis
can measure delay and false-trigger rate precisely.

This experiment is intentionally kept *separate* from the main organic-drift
benchmark.  It is a controlled calibration test, not a claim about real drift.

Usage
-----
  python scripts/run_known_changepoints.py [--n-post-cp 5] [--seeds 3]

Output
------
  results/changepoint_results.csv     – raw per-window results
  results/changepoint_summary.csv     – delay & FPR per policy
"""
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ChangepointExperiment")

# Ground-truth change-point window indices (zero-indexed)
CHANGE_POINTS = [5, 10]


def generate_changepoint_stream(
    n_pre: int = 5,
    n_post_cp: int = 5,
    samples_per_window: int = 1500,
    n_features: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a stream with two known abrupt concept change points.

    Structure:
      Phase A: windows 0..n_pre-1           (stable)
      Phase B: windows n_pre..n_pre+n_post-1    (after CP1)
      Phase C: windows n_pre+n_post..end        (after CP2)

    Drift is ABRUPT (step change, not gradual) so a good detector should
    identify it within 1-2 windows of the true boundary.

    Returns:
        pd.DataFrame with columns TransactionDT, isFraud, feature_0..N-1,
        plus a 'phase' column for ground-truth labelling.
    """
    rng = np.random.RandomState(seed)
    time_step = 86400.0 * 30  # monthly windows
    records = []

    # ---- Phase boundaries ----
    # Phase A: windows [0, n_pre)
    # Phase B: windows [n_pre, n_pre + n_post_cp)    — after CP1
    # Phase C: windows [n_pre + n_post_cp, total)    — after CP2
    n_total = n_pre + 2 * n_post_cp

    # Phase-specific covariate means (abrupt shift on CP)
    means_a = np.zeros(n_features)
    means_b = np.zeros(n_features)
    means_b[0] = 2.0   # strong mean shift on first feature
    means_b[2] = -2.0  # and third feature
    means_c = np.zeros(n_features)
    means_c[1] = 2.5   # different features shift in phase C
    means_c[3] = -1.5

    for w in range(n_total):
        if w < n_pre:
            phase = "A"
            means = means_a
            logit_coef = np.array([0.8, -0.5, 0.2] + [0.0] * (n_features - 3))
        elif w < n_pre + n_post_cp:
            phase = "B"
            means = means_b
            logit_coef = np.array([-0.6, 0.0, 0.7] + [0.0] * (n_features - 3))
        else:
            phase = "C"
            means = means_c
            logit_coef = np.array([0.0, 0.5, 0.0, 0.4] + [0.0] * (n_features - 4))

        cov = np.eye(n_features) * 0.5
        X_w = rng.multivariate_normal(means, cov, size=samples_per_window)
        logit = X_w @ logit_coef
        prob = 1.0 / (1.0 + np.exp(-logit))
        y_w = (rng.uniform(0, 1, size=samples_per_window) < prob).astype(int)

        w_start = w * time_step
        w_end = (w + 1) * time_step - 1.0
        timestamps = np.sort(rng.uniform(w_start, w_end, size=samples_per_window))

        for i in range(samples_per_window):
            row = {
                "TransactionDT": timestamps[i],
                "isFraud": y_w[i],
                "phase": phase,
                "window_id_gt": w,  # ground-truth window index
                "is_changepoint": int(w in CHANGE_POINTS),
            }
            for f in range(n_features):
                row[f"feature_{f}"] = float(X_w[i, f])
            records.append(row)

    df = pd.DataFrame(records)
    logger.info(
        f"Generated changepoint stream: {len(df):,} rows, {n_total} windows, "
        f"change points at windows {CHANGE_POINTS}."
    )
    return df


def compute_changepoint_metrics(
    results: pd.DataFrame,
    change_points: List[int],
    n_total_windows: int,
) -> pd.DataFrame:
    """Compute detection delay and false-trigger rate per policy.

    Args:
        results: Output DataFrame from ExperimentRunner.run_all()
        change_points: List of window indices where true change points occur.
        n_total_windows: Total number of windows in the stream.

    Returns:
        Summary DataFrame with one row per policy.
    """
    summary_rows = []

    # Windows that are "near" a change point (within 2 windows after it)
    detection_zone = set()
    for cp in change_points:
        for offset in range(1, 4):  # CP fires at cp, detected at cp+1..cp+3
            detection_zone.add(cp + offset)

    non_cp_windows = set(range(1, n_total_windows)) - detection_zone

    for policy_id in results["policy"].unique():
        p_rows = results[results["policy"] == policy_id].copy()
        p_rows = p_rows[p_rows["window_id"] > 0]  # skip window 0 (initial train)
        retrain_windows = set(p_rows[p_rows["retrained"] == True]["window_id"].tolist())

        # --- Detection delay for each change point ---
        delays = []
        for cp in change_points:
            # The model is evaluated on window cp (pre-retrain eval), so a
            # correct trigger should occur at cp or cp+1 at the latest.
            detected_at = None
            for offset in range(1, n_total_windows - cp):
                if (cp + offset) in retrain_windows:
                    detected_at = offset
                    break
            delays.append(detected_at if detected_at is not None else float("nan"))

        # --- False-trigger rate: retrains in non-CP windows ---
        fp_retrains = len(retrain_windows & non_cp_windows)
        fp_rate = fp_retrains / len(non_cp_windows) if non_cp_windows else 0.0

        summary_rows.append({
            "policy": policy_id,
            "cp1_detection_delay": delays[0] if len(delays) > 0 else float("nan"),
            "cp2_detection_delay": delays[1] if len(delays) > 1 else float("nan"),
            "mean_detection_delay": float(np.nanmean(delays)),
            "missed_changepoints": sum(1 for d in delays if np.isnan(d)),
            "false_trigger_retrains": fp_retrains,
            "false_trigger_rate": round(fp_rate, 4),
            "total_retrains": len(retrain_windows),
        })

    return pd.DataFrame(summary_rows)


def run_changepoint_experiment(n_post_cp: int = 5, n_seeds: int = 3) -> pd.DataFrame:
    """Run the known change-point experiment and print results."""
    config = {
        "project": {"name": "changepoint_control", "seed": 42, "n_seeds": 1},
        "data": {
            "dataset": "fraud",
            "timestamp_col": "TransactionDT",
            "target_col": "isFraud",
            "window_size": "monthly",
            "min_window_samples": 500,
            "data_source_type": "synthetic_controlled",
        },
        "model": {
            "type": "xgboost",
            "params": {"n_estimators": 100, "max_depth": 5, "eval_metric": "logloss"},
        },
        "detection": {"statistical_method": "ks", "ks_alpha": 0.05},
        "explainability": {
            "shap_method": "tree",
            "max_shap_samples": 200,
            "magnitude_threshold": 0.030,
            "rank_change_threshold": 0.035,
        },
        "fusion": {
            "alpha": 0.4, "beta": 0.4, "gamma": 0.2,
            "adaptive_lookback": 3, "adaptive_lambda": 1.5,
        },
        "policy": {
            "active_policies": [
                "p0_never", "p1_fixed", "p2_drift_only",
                "p3_shap_magnitude", "p4_shap_rank", "p5_dis_fused",
            ],
            "fixed_schedule_interval": 3,
        },
        "evaluation": {"metrics": ["accuracy"], "cost_metric": "retrain_count"},
        "output": {"results_dir": str(PROJECT_ROOT / "results")},
    }

    n_pre = 5
    n_total = n_pre + 2 * n_post_cp
    all_results = []

    for seed in range(42, 42 + n_seeds):
        df_cp = generate_changepoint_stream(
            n_pre=n_pre, n_post_cp=n_post_cp,
            samples_per_window=1500, n_features=15, seed=seed,
        )
        runner = ExperimentRunner(config)
        seed_results = runner.run_all(
            df_override=df_cp,
            output_filename=f"changepoint_seed{seed}.csv",
        )
        seed_results["cp_seed"] = seed
        all_results.append(seed_results)

    results = pd.concat(all_results, ignore_index=True)

    # Save full results
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    full_path = out_dir / "changepoint_results.csv"
    results.to_csv(full_path, index=False)

    # Compute per-seed metrics, then average
    all_summaries = []
    for seed in range(42, 42 + n_seeds):
        seed_results = results[results["cp_seed"] == seed]
        s = compute_changepoint_metrics(
            seed_results, change_points=CHANGE_POINTS, n_total_windows=n_total
        )
        s["seed"] = seed
        all_summaries.append(s)

    summary_all = pd.concat(all_summaries)
    summary = (
        summary_all.groupby("policy")
        .agg({
            "cp1_detection_delay": "mean",
            "cp2_detection_delay": "mean",
            "mean_detection_delay": "mean",
            "missed_changepoints": "mean",
            "false_trigger_retrains": "mean",
            "false_trigger_rate": "mean",
            "total_retrains": "mean",
        })
        .round(3)
        .reset_index()
    )
    summary.columns = [f"mean_{c}" if c != "policy" else c for c in summary.columns]
    summary_path = out_dir / "changepoint_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("KNOWN CHANGE-POINT DETECTION EXPERIMENT")
    print(f"Change points at windows: {CHANGE_POINTS}")
    print(f"Stream: {n_total} windows ({n_pre} stable + {n_post_cp}×2 post-CP)")
    print(f"Seeds: {n_seeds}")
    print("=" * 70)
    print(f"\n{summary.to_string(index=False)}")
    print("\nInterpretation:")
    print("  mean_cp*_detection_delay: windows after CP before first retrain")
    print("  (lower is better; NaN = change point never detected = miss)")
    print("  mean_false_trigger_rate: fraction of non-CP windows that triggered retrain")
    print("  (lower is better for drift-responsive policies; P1 is by design)")
    print(f"\nFull results: {full_path}")
    print(f"Summary:      {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="DRIFT-X Known Change-Point Detection Experiment"
    )
    parser.add_argument(
        "--n-post-cp", type=int, default=5,
        help="Windows per phase after each change point (default: 5)"
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of random seeds (default: 3)"
    )
    args = parser.parse_args()
    run_changepoint_experiment(n_post_cp=args.n_post_cp, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
