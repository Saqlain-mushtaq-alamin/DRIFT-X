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
    seed: int = 101,
    null_type: str = "independent",
) -> pd.DataFrame:
    """Generate a stationary (zero-drift) dataset under various correlation/tail structures.

    Args:
        n_windows: Number of windows to generate.
        samples_per_window: Number of rows per window.
        n_features: Number of features.
        seed: Random seed.
        null_type: 'independent', 'correlated', 'autocorrelated', 'heavy_tailed', or 'elec2_null'.
    """
    if null_type == "elec2_null":
        elec_path = PROJECT_ROOT / "data" / "raw" / "electricity.csv"
        if elec_path.exists():
            df_raw = pd.read_csv(elec_path)
            slice_df = df_raw.head(3000).copy()
            target_col = "class" if "class" in slice_df.columns else "target"
            feature_cols = [c for c in slice_df.columns if c not in [target_col, "date", "day", "period"] and np.issubdtype(slice_df[c].dtype, np.number)]
            rng = np.random.RandomState(seed)
            records = []
            time_step = 86400.0 * 30
            for w in range(n_windows):
                idx = rng.choice(len(slice_df), size=samples_per_window, replace=True)
                sample = slice_df.iloc[idx].reset_index(drop=True)
                y_w = (sample[target_col].astype(str).str.upper() == "UP").astype(int)
                w_start = w * time_step
                w_end = (w + 1) * time_step - 1.0
                timestamps = np.sort(rng.uniform(w_start, w_end, size=samples_per_window))
                for i in range(samples_per_window):
                    row = {"TransactionDT": timestamps[i], "isFraud": int(y_w.iloc[i])}
                    for f_idx, col in enumerate(feature_cols[:n_features]):
                        row[f"feature_{f_idx}"] = float(sample[col].iloc[i])
                    records.append(row)
            df = pd.DataFrame(records)
            logger.info(f"Generated elec2_null stream: {len(df):,} rows from stationary ELEC2 slice.")
            return df

    rng = np.random.RandomState(seed)
    means = rng.uniform(-0.5, 0.5, size=n_features)
    time_step = 86400.0 * 30

    if null_type == "correlated":
        # Toeplitz correlation matrix across features
        cov = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                cov[i, j] = 0.5 * (0.6 ** abs(i - j))
    else:
        cov = np.eye(n_features) * 0.5

    # Concept coefficients: 4 informative features
    concept_coef = np.zeros(n_features)
    concept_coef[0] = 2.0
    concept_coef[1] = -1.5
    concept_coef[2] = 1.2
    concept_coef[3] = -1.0

    records = []
    for w in range(n_windows):
        if null_type == "heavy_tailed":
            # Student-t with df=4 (heavy tails)
            z = rng.standard_t(df=4, size=(samples_per_window, n_features))
            X_w = means + z * np.sqrt(0.5 * (4 - 2) / 4)
        elif null_type == "autocorrelated":
            # AR(1) autocorrelation across samples within window (phi=0.4)
            X_w = np.zeros((samples_per_window, n_features))
            noise = rng.multivariate_normal(np.zeros(n_features), cov, size=samples_per_window)
            phi = 0.4
            scale = np.sqrt(1.0 - phi**2)
            cur = noise[0]
            for t in range(samples_per_window):
                cur = phi * cur + scale * noise[t]
                X_w[t] = means + cur
        else:
            X_w = rng.multivariate_normal(means, cov, size=samples_per_window)

        logit = X_w @ concept_coef
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
        f"Generated {null_type} null stream: {len(df):,} rows, {n_windows} windows, "
        f"{n_features} features — NO DRIFT"
    )
    return df


def run_null_stream_experiment(
    n_windows: int = 20,
    n_seeds: int = 5,
    null_type: str = "independent",
    seed_start: int = 101,
) -> pd.DataFrame:
    """Run all 10 policies on the null stream and report false-positive retrain counts."""
    config = {
        "project": {"name": f"null_stream_{null_type}", "seed": seed_start, "n_seeds": n_seeds},
        "data": {
            "dataset": "fraud",
            "timestamp_col": "TransactionDT",
            "target_col": "isFraud",
            "window_size": "monthly",
            "min_window_samples": 500,
            "data_source_type": f"synthetic_null_{null_type}",
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
            "use_calibrated_zscore": True,
            "clip_max": 5.0,
            "gating_mode": "two_channel",
            "gating_channel_threshold": 2.5,
            "gating_min_channels": 2,
            "threshold_mode": "fixed",
            "fixed_threshold": 3.0,
            "warmup_threshold": float("inf"),
            "adaptive_lookback": 10,
            "adaptive_lambda": 1.92,
        },
        "policy": {
            "active_policies": [
                "p0_never", "p1_fixed", "p2_drift_only",
                "p3_shap_magnitude", "p4_shap_rank", "p5_dis_fused",
                "p6_performance_drop", "p7_weighted_ks", "p8_random_budget", "p9_shap_loss",
            ],
            "fixed_schedule_interval": 3,
        },
        "evaluation": {"metrics": ["accuracy"], "cost_metric": "retrain_count"},
        "output": {"results_dir": str(PROJECT_ROOT / "results")},
    }

    all_rows = []
    seeds = [seed_start + i for i in range(n_seeds)]
    for seed in seeds:
        df_null = generate_null_stream(
            n_windows=n_windows,
            samples_per_window=1500,
            n_features=15,
            seed=seed,
            null_type=null_type,
        )
        runner = ExperimentRunner(config)
        seed_results = runner.run_all(
            df_override=df_null,
            output_filename=f"null_stream_{null_type}_seed{seed}.csv",
        )
        seed_results["null_seed"] = seed
        seed_results["null_type"] = null_type
        all_rows.append(seed_results)

    results = pd.concat(all_rows, ignore_index=True)

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    full_path = out_dir / f"null_stream_{null_type}_results.csv"
    results.to_csv(full_path, index=False)
    if null_type == "independent":
        results.to_csv(out_dir / "null_stream_results.csv", index=False)

    eval_rows = results[results["window_id"] > 0].copy()
    summary_rows = []

    for policy_id in results["policy"].unique():
        p_rows = eval_rows[eval_rows["policy"] == policy_id]
        retrain_rows = p_rows[p_rows["retrained"] == True]
        n_false_positives = len(retrain_rows)
        n_total_windows = len(p_rows)
        fp_rate = n_false_positives / n_total_windows if n_total_windows > 0 else 0.0

        mean_acc = p_rows["accuracy"].dropna().mean()

        summary_rows.append({
            "policy": policy_id,
            "false_positive_retrains": n_false_positives,
            "total_eval_windows": n_total_windows,
            "false_positive_rate": round(fp_rate, 4),
            "mean_accuracy_null": round(mean_acc, 4) if not np.isnan(mean_acc) else float("nan"),
            "expected_retrains_ideal": 0,
            "null_type": null_type,
            "note": (
                "by_design" if policy_id in ["p1_fixed", "p8_random_budget"]
                else ("never_retrains" if policy_id == "p0_never" else "")
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"null_stream_{null_type}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    if null_type == "independent":
        summary_df.to_csv(out_dir / "null_stream_summary.csv", index=False)

    print("\n" + "=" * 70)
    print(f"NULL STREAM CONTROL EXPERIMENT RESULTS ({null_type.upper()})")
    print("(Stationary data — zero drift — false positives are evaluation failures)")
    print("=" * 70)
    print(f"\nWindows per run: {n_windows}  |  Seeds: {n_seeds}")
    print(f"\n{summary_df.to_string(index=False)}")
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
    parser.add_argument(
        "--seed-start", type=int, default=101,
        help="Starting seed (default: 101, distinct from training/calibration seeds)"
    )
    parser.add_argument(
        "--null-type", type=str, default="independent",
        choices=["independent", "correlated", "autocorrelated", "heavy_tailed", "elec2_null"],
        help="Type of null stream distribution"
    )
    args = parser.parse_args()

    run_null_stream_experiment(
        n_windows=args.n_windows,
        n_seeds=args.seeds,
        null_type=args.null_type,
        seed_start=args.seed_start,
    )


if __name__ == "__main__":
    main()
