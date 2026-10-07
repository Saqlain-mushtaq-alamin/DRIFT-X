"""Null-stream control experiment for DRIFT-X.

Purpose
-------
A correctly calibrated drift trigger should retrain approximately ZERO times
when no drift is actually present in the data.  This script verifies that
property for every policy.

What it tests
-------------
Generates a stationary data stream (no feature or concept drift at all) and
runs the full DRIFT-X pipeline on it.  Any retrains observed beyond the
mandatory window-0 initial training are *false positives*.  The expected
retrain count for a well-calibrated detector on stationary data is:

  P0 (never):      0 additional retrains (correct by design)
  P1 (fixed):      ceil(N/interval) - 1  (correct by design — it ignores drift)
  P2 (drift-only): ~0  (KS should not reject H0 on iid data)
  P3 (shap-mag):   ~0  (SHAP profiles stable on stationary data)
  P4 (shap-rank):  ~0
  P5 (DIS-fused):  ~0

Failure modes to look for
--------------------------
- DIS fires at local extremes even when nothing drifts.  This happens because
  the SignalNormalizer rescales by running min-max: the *current* value is
  always the max or min at some point, causing a normalised score of 1.0 that
  can exceed the ATC threshold.  The ATC warmup and lambda=1.5 should suppress
  most of these, but the null stream quantifies the residual false-positive rate.

Usage
-----
  python scripts/run_null_stream.py [--n-windows 20] [--seeds 5]

Output
------
  results/null_stream_results.csv   – raw per-window per-policy results
  results/null_stream_summary.csv   – retrain counts per policy
  (Printed to stdout as well)
"""
import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner

logging.basicConfig(
    level=logging.WARNING,  # suppress per-window chatter
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("NullStream")


def generate_null_stream(
    n_windows: int = 20,
    samples_per_window: int = 1500,
    n_features: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a stationary (zero-drift) dataset.

    Feature distributions and the decision boundary are IDENTICAL across all
    windows.  There is no concept drift, no covariate shift.  Timestamps are
    monotonically increasing so the windower treats them as temporal windows.

    Returns:
        pd.DataFrame with columns TransactionDT, isFraud, feature_0..feature_N-1
    """
    rng = np.random.RandomState(seed)
    means = rng.uniform(-0.5, 0.5, size=n_features)
    cov = np.eye(n_features) * 0.25  # moderate variance, no covariance
    time_step = 86400.0 * 30  # 30-day windows (monthly)

    records = []
    for w in range(n_windows):
        X_w = rng.multivariate_normal(means, cov, size=samples_per_window)
        # Fixed logit: same coefficients every window
        logit = X_w[:, 0] * 0.8 - X_w[:, 1] * 0.5 + X_w[:, 2] * 0.2
        prob = 1.0 / (1.0 + np.exp(-logit))
        y_w = (rng.uniform(0, 1, size=samples_per_window) < prob).astype(int)

        w_start = w * time_step
        w_end = (w + 1) * time_step - 1.0
        timestamps = np.sort(rng.uniform(w_start, w_end, size=samples_per_window))

        for i in range(samples_per_window):
            row = {"TransactionDT": timestamps[i], "isFraud": y_w[i]}
            for f in range(n_features):
                row[f"feature_{f}"] = X_w[i, f]
            records.append(row)

    df = pd.DataFrame(records)
    logger.info(
        f"Generated null stream: {len(df):,} rows, {n_windows} windows, "
        f"{n_features} features — NO DRIFT"
    )
    return df


def run_null_stream_experiment(n_windows: int = 20, n_seeds: int = 5) -> pd.DataFrame:
    """Run all policies on the null stream and report false-positive retrain counts."""
    config = {
        "project": {"name": "null_stream_control", "seed": 42, "n_seeds": n_seeds},
        "data": {
            "dataset": "fraud",
            "timestamp_col": "TransactionDT",
            "target_col": "isFraud",
            "window_size": "monthly",
            "min_window_samples": 500,
            # Explicitly mark as synthetic/null for traceability
            "data_source_type": "synthetic_null",
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

    # Generate stationary data (no drift) for multiple seeds to get distribution
    all_rows = []
    seeds = [42 + i for i in range(n_seeds)]
    for seed in seeds:
        df_null = generate_null_stream(
            n_windows=n_windows,
            samples_per_window=1500,
            n_features=15,
            seed=seed,
        )
        runner = ExperimentRunner(config)
        seed_results = runner.run_all(
            df_override=df_null,
            output_filename=f"null_stream_seed{seed}.csv",
        )
        seed_results["null_seed"] = seed
        all_rows.append(seed_results)

    results = pd.concat(all_rows, ignore_index=True)

    # Save full results
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    full_path = out_dir / "null_stream_results.csv"
    results.to_csv(full_path, index=False)

    # -----------------------------------------------------------------------
    # Summary: false-positive retrains per policy (beyond mandatory window-0)
    # -----------------------------------------------------------------------
    # Exclude window 0 (always retrained) and count additional retrains
    eval_rows = results[results["window_id"] > 0].copy()
    summary_rows = []

    for policy_id in results["policy"].unique():
        p_rows = eval_rows[eval_rows["policy"] == policy_id]
        retrain_rows = p_rows[p_rows["retrained"] == True]
        n_false_positives = len(retrain_rows)
        n_total_windows = len(p_rows)
        fp_rate = n_false_positives / n_total_windows if n_total_windows > 0 else 0.0

        # Mean accuracy (should be ~stable if no drift and no false retrains)
        mean_acc = p_rows["accuracy"].dropna().mean()

        summary_rows.append({
            "policy": policy_id,
            "false_positive_retrains": n_false_positives,
            "total_eval_windows": n_total_windows,
            "false_positive_rate": round(fp_rate, 4),
            "mean_accuracy_null": round(mean_acc, 4) if not np.isnan(mean_acc) else float("nan"),
            "expected_retrains_ideal": 0,
            "note": (
                "by_design" if policy_id == "p1_fixed"
                else ("never_retrains" if policy_id == "p0_never" else "")
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "null_stream_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Print results
    print("\n" + "=" * 70)
    print("NULL STREAM CONTROL EXPERIMENT RESULTS")
    print("(Stationary data — zero drift — false positives are evaluation failures)")
    print("=" * 70)
    print(f"\nWindows per run: {n_windows}  |  Seeds: {n_seeds}")
    print(f"\n{summary_df.to_string(index=False)}")
    print("\nInterpretation:")
    print("  P0: 0 retrains expected (never-retrain baseline)")
    print("  P1: retrains every 3 windows by design (not a false positive — it ignores drift)")
    print("  P2-P5: false_positive_retrains should be ~0 on stationary data.")
    print("  Non-zero values for P2-P5 indicate detector sensitivity issues.")
    print(f"\nFull results: {full_path}")
    print(f"Summary:      {summary_path}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="DRIFT-X Null-Stream False-Positive Control Experiment"
    )
    parser.add_argument(
        "--n-windows", type=int, default=20,
        help="Number of stationary windows to generate (default: 20)"
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of random seeds (default: 5)"
    )
    args = parser.parse_args()

    run_null_stream_experiment(n_windows=args.n_windows, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
