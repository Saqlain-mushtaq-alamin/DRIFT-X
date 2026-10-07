"""Hyperparameter tuning on a held-out stream — DRIFT-X.

Purpose
-------
Tunes the six DRIFT-X hyperparameters:

  alpha, beta, gamma  — DIS fusion weights
  magnitude_threshold — P3 SHAP-magnitude retrain threshold
  rank_change_threshold — P4 SHAP rank-change retrain threshold
  adaptive_lambda (λ) — ATC sensitivity multiplier
  adaptive_lookback (k) — ATC rolling-window size

on a SEPARATE stream that is NEVER used in the final evaluation.  The chosen
values are written to ``configs/frozen.yaml``.  The main experiment scripts
should be pointed at ``frozen.yaml`` instead of ``default.yaml`` so that the
final evaluation is a single, unmodified run on untouched data.

Two tuning sources are supported (choose via ``--source``):
  synthetic  (default)
    A synthetic replicate generated with a DIFFERENT seed from the evaluation
    streams.  Safe to use in CI and offline development.
  elec2_head
    The first N windows of the real ELEC2 dataset (early slice only — the
    evaluation uses the full stream, so the head slice is disjoint if you
    use a strict temporal split).  Requires ``data/raw/electricity.csv`` to
    exist (OpenML download).  Will refuse to run if the file has fewer than
    45,312 rows (synthetic fallback detected).

Usage
-----
  # Tune on synthetic holdout (default, offline-safe):
  python scripts/tune_on_holdout.py --source synthetic --seeds 3

  # Tune on the first 10 windows of ELEC2 (requires real ELEC2 download):
  python scripts/tune_on_holdout.py --source elec2_head --head-windows 10

Output
------
  configs/frozen.yaml   — frozen hyperparameters to use for final evaluation
  results/tuning_log.csv — full grid results on the holdout stream
"""
import sys
import copy
import logging
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("TuneOnHoldout")

# ---------------------------------------------------------------------------
# Search grid — deliberately small to stay fast.  Extend if you have time.
# ---------------------------------------------------------------------------
ALPHA_BETA_GAMMA_GRIDS = [
    # (alpha, beta, gamma)  — must sum to 1.0
    (0.4, 0.4, 0.2),   # default
    (0.5, 0.3, 0.2),   # up-weight stat
    (0.3, 0.5, 0.2),   # up-weight SHAP magnitude
    (0.33, 0.33, 0.34), # equal weights
    (0.5, 0.4, 0.1),   # down-weight rank
]
LAMBDA_VALUES = [1.0, 1.5, 2.0, 2.5]
LOOKBACK_VALUES = [2, 3, 5]
MAG_THRESHOLDS = [0.020, 0.030, 0.045]
RANK_THRESHOLDS = [0.025, 0.035, 0.050]

# Tuning objective: maximise P5 mean accuracy on the holdout stream.
# A secondary tiebreaker is fewer retrains (prefer parsimonious triggers).
OBJECTIVE_POLICY = "p5_dis_fused"


# ---------------------------------------------------------------------------
# Holdout stream generators
# ---------------------------------------------------------------------------

def load_synthetic_holdout(seed: int = 999, n_windows: int = 20) -> pd.DataFrame:
    """Generate a synthetic holdout stream with a seed NOT used in evaluation.

    The evaluation streams use seeds 42–46.  The default holdout seed is 999
    to ensure complete independence.
    """
    from scripts.download_data import generate_synthetic_drift_data
    out_path = PROJECT_ROOT / "data" / "raw" / f"holdout_seed{seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_drift_data(
        output_path=str(out_path),
        n_windows=n_windows,
        samples_per_window=1500,
        n_features=15,
        seed=seed,
        drift_intensity=0.4,
    )
    logger.info(f"Loaded synthetic holdout: {len(df):,} rows, seed={seed}")
    return df


def load_elec2_head(head_windows: int = 10) -> pd.DataFrame:
    """Load only the first ``head_windows`` temporal windows of ELEC2.

    This is the held-out tuning slice.  The evaluation uses the *full* ELEC2
    stream (all 31 monthly windows), so there is no information leak as long
    as you do not look at the full-stream results before freezing the config.

    Requires data/raw/electricity.csv (real OpenML download).
    """
    ELEC2_PATH = PROJECT_ROOT / "data" / "raw" / "electricity.csv"
    if not ELEC2_PATH.exists():
        raise FileNotFoundError(
            f"ELEC2 file not found at {ELEC2_PATH}. "
            "Run: python scripts/download_data.py --dataset electricity"
        )
    df = pd.read_csv(ELEC2_PATH)
    ELEC2_EXPECTED = 45_312
    if len(df) != ELEC2_EXPECTED:
        raise AssertionError(
            f"ELEC2 file has {len(df)} rows, expected {ELEC2_EXPECTED}. "
            "This looks like the synthetic fallback. Download the real dataset first."
        )
    # Sort by timestamp and take the first head_windows monthly chunks
    df = df.sort_values("Timestamp").reset_index(drop=True)
    # Monthly step: approx 1440 rows per window (45,312 / 31 months)
    rows_per_window = len(df) // 31
    cutoff = head_windows * rows_per_window
    df_head = df.iloc[:cutoff].copy()
    logger.info(
        f"Loaded ELEC2 head slice: {len(df_head):,} rows ({head_windows} windows)"
    )
    return df_head


# ---------------------------------------------------------------------------
# Tuning engine
# ---------------------------------------------------------------------------

def _make_config(alpha, beta, gamma, lam, k, mag_thr, rank_thr,
                 ts_col, tgt_col, window_size, tmp_dir) -> dict:
    return {
        "project": {"name": "tuning", "seed": 42, "n_seeds": 1},
        "data": {
            "dataset": "fraud" if ts_col == "TransactionDT" else "electricity",
            "timestamp_col": ts_col,
            "target_col": tgt_col,
            "window_size": window_size,
            "min_window_samples": 500,
            "data_source_type": "synthetic",
        },
        "model": {
            "type": "xgboost",
            "params": {"n_estimators": 100, "max_depth": 5, "eval_metric": "logloss"},
        },
        "detection": {"statistical_method": "ks", "ks_alpha": 0.05},
        "explainability": {
            "shap_method": "tree",
            "max_shap_samples": 200,
            "magnitude_threshold": mag_thr,
            "rank_change_threshold": rank_thr,
        },
        "fusion": {
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "adaptive_lookback": k, "adaptive_lambda": lam,
        },
        "policy": {
            "active_policies": [OBJECTIVE_POLICY],
            "fixed_schedule_interval": 3,
        },
        "evaluation": {"metrics": ["accuracy"], "cost_metric": "retrain_count"},
        "output": {"results_dir": str(tmp_dir)},
    }


def tune(
    df_holdout: pd.DataFrame,
    ts_col: str,
    tgt_col: str,
    window_size,
    n_seeds: int = 1,
) -> dict:
    """Run the hyperparameter grid on ``df_holdout`` and return the best config."""
    import tempfile, os

    grid_records = []

    combos = list(product(
        ALPHA_BETA_GAMMA_GRIDS,
        LAMBDA_VALUES,
        LOOKBACK_VALUES,
        MAG_THRESHOLDS,
        RANK_THRESHOLDS,
    ))
    print(f"  Grid size: {len(combos)} combinations × {n_seeds} seed(s) "
          f"= {len(combos) * n_seeds} runs")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        for i, ((alpha, beta, gamma), lam, k, mag_thr, rank_thr) in enumerate(combos):
            cfg = _make_config(alpha, beta, gamma, lam, k, mag_thr, rank_thr,
                               ts_col, tgt_col, window_size, tmp_dir)
            cfg["project"]["n_seeds"] = n_seeds

            runner = ExperimentRunner(cfg)
            results = runner.run_all(
                df_override=df_holdout.copy(),
                output_filename=f"tune_run_{i}.csv",
            )

            # Evaluate on window 1+ only (window 0 = NaN by design)
            eval_rows = results[results["window_id"] > 0]
            mean_acc = eval_rows["accuracy"].mean()
            mean_retrains = results.groupby("seed")["retrain_count"].last().mean() - 1

            grid_records.append({
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "lambda": lam, "k": k,
                "mag_threshold": mag_thr,
                "rank_threshold": rank_thr,
                "mean_accuracy": round(float(mean_acc), 6),
                "mean_retrains_excl_init": round(float(mean_retrains), 2),
            })

            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{len(combos)} grid points...")

    grid_df = pd.DataFrame(grid_records)
    # Primary: maximise accuracy; secondary: minimise retrains (tiebreaker)
    grid_df = grid_df.sort_values(
        ["mean_accuracy", "mean_retrains_excl_init"],
        ascending=[False, True],
    ).reset_index(drop=True)

    best = grid_df.iloc[0].to_dict()
    return best, grid_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DRIFT-X Holdout Hyperparameter Tuning"
    )
    parser.add_argument(
        "--source", choices=["synthetic", "elec2_head"], default="synthetic",
        help="Holdout data source: 'synthetic' (safe, offline) or 'elec2_head' "
             "(real ELEC2 early slice, requires download). Default: synthetic"
    )
    parser.add_argument(
        "--holdout-seed", type=int, default=999,
        help="Seed for synthetic holdout generator. Must differ from evaluation "
             "seeds (42–46). Default: 999"
    )
    parser.add_argument(
        "--holdout-windows", type=int, default=20,
        help="Number of synthetic holdout windows. Default: 20"
    )
    parser.add_argument(
        "--head-windows", type=int, default=10,
        help="Number of ELEC2 head windows (elec2_head source only). Default: 10"
    )
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Seeds per grid point. Higher = more stable but slower. Default: 1"
    )
    args = parser.parse_args()

    # Warn if holdout seed overlaps with evaluation seeds
    EVAL_SEEDS = set(range(42, 47))
    if args.source == "synthetic" and args.holdout_seed in EVAL_SEEDS:
        print(f"[WARNING] --holdout-seed {args.holdout_seed} overlaps with "
              f"evaluation seeds {sorted(EVAL_SEEDS)}. Use a different seed "
              "to ensure independence (recommended: 999).")

    print("=" * 65)
    print("DRIFT-X HOLDOUT HYPERPARAMETER TUNING")
    print("=" * 65)

    # --- Load holdout data ---
    if args.source == "synthetic":
        print(f"\nSource: synthetic holdout (seed={args.holdout_seed}, "
              f"windows={args.holdout_windows})")
        df_holdout = load_synthetic_holdout(
            seed=args.holdout_seed, n_windows=args.holdout_windows
        )
        ts_col, tgt_col, window_size = "TransactionDT", "isFraud", "monthly"
    else:
        print(f"\nSource: ELEC2 head slice (first {args.head_windows} windows)")
        df_holdout = load_elec2_head(head_windows=args.head_windows)
        ts_col, tgt_col, window_size = "Timestamp", "target", "monthly"

    print(f"Holdout: {len(df_holdout):,} rows")
    print(f"\nRunning grid search...")

    best, grid_df = tune(
        df_holdout, ts_col=ts_col, tgt_col=tgt_col,
        window_size=window_size, n_seeds=args.seeds,
    )

    # --- Save tuning log ---
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / "tuning_log.csv"
    grid_df.to_csv(log_path, index=False)
    print(f"\nTuning log saved: {log_path}")

    # --- Write frozen.yaml ---
    # Load default config and override with best values
    default_cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(default_cfg_path, "r", encoding="utf-8") as f:
        frozen = yaml.safe_load(f)

    frozen["fusion"]["alpha"] = float(best["alpha"])
    frozen["fusion"]["beta"] = float(best["beta"])
    frozen["fusion"]["gamma"] = float(best["gamma"])
    frozen["fusion"]["adaptive_lambda"] = float(best["lambda"])
    frozen["fusion"]["adaptive_lookback"] = int(best["k"])
    frozen["explainability"]["magnitude_threshold"] = float(best["mag_threshold"])
    frozen["explainability"]["rank_change_threshold"] = float(best["rank_threshold"])

    # Annotate provenance
    frozen["_tuning_provenance"] = {
        "source": args.source,
        "holdout_seed": args.holdout_seed if args.source == "synthetic" else "elec2_head",
        "holdout_windows": args.holdout_windows if args.source == "synthetic" else args.head_windows,
        "best_holdout_accuracy": round(float(best["mean_accuracy"]), 6),
        "best_holdout_retrains": round(float(best["mean_retrains_excl_init"]), 2),
        "note": (
            "Parameters frozen from holdout tuning. "
            "Do NOT re-tune on evaluation data. "
            "Use this file as --config for all final experiment runs."
        ),
    }

    frozen_path = PROJECT_ROOT / "configs" / "frozen.yaml"
    with open(frozen_path, "w", encoding="utf-8") as f:
        yaml.dump(frozen, f, default_flow_style=False, sort_keys=False)

    print(f"Frozen config saved: {frozen_path}")

    print("\n" + "=" * 65)
    print("BEST HYPERPARAMETERS (from holdout tuning)")
    print("=" * 65)
    for k_name, v in best.items():
        print(f"  {k_name:30s}: {v}")
    print()
    print("Next step:")
    print(f"  python scripts/run_experiment.py --config configs/frozen.yaml")
    print("  (Do NOT re-run tune_on_holdout.py after seeing evaluation results.)")

    return best


if __name__ == "__main__":
    main()
