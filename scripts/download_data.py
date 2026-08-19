"""Dataset download and synthetic data generation script for DRIFT-X."""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_drift_data(
    output_path: str,
    n_windows: int = 20,
    samples_per_window: int = 1500,
    n_features: int = 15,
    seed: int = 42,
    drift_intensity: float = 0.4,
) -> pd.DataFrame:
    """
    Generate synthetic dataset with realistic organic feature and concept drift
    across temporal windows.

    Args:
        output_path: Path to write the output CSV file.
        n_windows: Number of temporal windows to generate (>=20 recommended for
            publication-grade experiments — gives >=15 windows in adaptive mode
            when lookback=3).
        samples_per_window: Rows per window.
        n_features: Total number of feature columns.
        seed: Base random seed.  Per-window noise is added on top so that
            different ``seed`` values produce genuinely different datasets.
        drift_intensity: Controls how strongly feature means drift over time.
            0.0 = no drift, 1.0 = very strong drift.  Default 0.4 is enough
            to reliably trigger the KS detector.

    Returns:
        Generated DataFrame (also written to ``output_path``).
    """
    np.random.seed(seed)
    records = []

    # Base feature distributions
    means = np.random.uniform(-1.0, 1.0, size=n_features)
    cov = np.eye(n_features)

    time_step = 86400.0 * 30  # ~30 days per window step (monthly)

    for w in range(n_windows):
        # Copy means and introduce gradual organic drift after window 2
        w_means = means.copy()
        if w >= 2:
            # Five features drift with different magnitudes — more features than
            # the old 3-feature version so KS detection fires more reliably.
            drifting_idx = [0, 2, 4, 6, 8]
            drift_magnitudes = np.array([0.4, -0.5, 0.35, -0.3, 0.45])
            w_means[drifting_idx] += (w - 1) * drift_magnitudes * drift_intensity

        # Per-window noise breaks cross-seed determinism so std > 0 after seed fix.
        w_means += np.random.normal(0, 0.05, size=n_features)

        # Sample features for this window
        X_w = np.random.multivariate_normal(w_means, cov, size=samples_per_window)

        # Three-phase concept drift so the model has real degradation to detect.
        if w < 5:
            logit = X_w[:, 0] * 0.8 - X_w[:, 1] * 0.5 + X_w[:, 2] * 0.2
        elif w < 12:
            logit = X_w[:, 0] * 0.3 - X_w[:, 1] * 0.8 + X_w[:, 4] * 0.9
        else:
            logit = -X_w[:, 0] * 0.6 + X_w[:, 2] * 0.7 + X_w[:, 6] * 0.5

        prob = 1.0 / (1.0 + np.exp(-logit))
        y_w = (np.random.uniform(0, 1, size=samples_per_window) < prob).astype(int)

        # Timestamps monotonically increasing within window interval
        w_start = w * time_step
        w_end = (w + 1) * time_step - 1.0
        timestamps = np.sort(np.random.uniform(w_start, w_end, size=samples_per_window))

        for i in range(samples_per_window):
            row = {"TransactionDT": timestamps[i], "isFraud": y_w[i]}
            for f in range(n_features):
                row[f"feature_{f}"] = X_w[i, f]
            records.append(row)

    df = pd.DataFrame(records)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"Generated synthetic drift dataset ({len(df):,} rows, {n_windows} windows) "
        f"-> {output_path}"
    )
    return df


def download_fraud_dataset(output_dir: str):
    """Download IEEE-CIS Fraud Detection dataset from Kaggle API or prompt manual download."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files('ieee-fraud-detection', path=output_dir)
        print(f"Downloaded IEEE-CIS dataset to {output_dir}")
    except Exception as e:
        print(f"Kaggle API unavailable or unauthenticated: {e}")
        print("Generating synthetic organic-drift dataset for development...")
        generate_synthetic_drift_data(os.path.join(output_dir, "synthetic_drift.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRIFT-X Data Downloader")
    parser.add_argument("--dataset", default="synthetic", choices=["fraud", "intrusion", "synthetic"])
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--n-windows", type=int, default=20, help="Windows to generate (synthetic only)")
    parser.add_argument("--drift-intensity", type=float, default=0.4, help="Drift strength (synthetic only)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset == "fraud":
        download_fraud_dataset(args.output_dir)
    elif args.dataset == "synthetic":
        generate_synthetic_drift_data(
            output_path=os.path.join(args.output_dir, "synthetic_drift.csv"),
            n_windows=args.n_windows,
            drift_intensity=args.drift_intensity,
        )
