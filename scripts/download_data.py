"""Dataset download and synthetic data generation script for DRIFT-X."""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_drift_data(
    output_path: str,
    n_windows: int = 10,
    samples_per_window: int = 2000,
    n_features: int = 15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic dataset with realistic organic feature and concept drift across temporal windows.
    """
    np.random.seed(seed)
    records = []
    
    # Base feature distributions
    means = np.random.uniform(-1.0, 1.0, size=n_features)
    cov = np.eye(n_features)
    
    current_time = 0.0
    time_step = 86400.0 * 30  # ~30 days per window step (monthly)
    
    for w in range(n_windows):
        # Copy means and introduce gradual organic drift after window 2
        w_means = means.copy()
        if w >= 2:
            drifting_idx = [0, 2, 4]  # Drifting features
            w_means[drifting_idx] += (w - 1) * np.array([0.25, -0.3, 0.2])
        
        # Sample features for this window
        X_w = np.random.multivariate_normal(w_means, cov, size=samples_per_window)
        
        # Concept drift: probability of target depends on features and window
        logit = X_w[:, 0] * 0.8 - X_w[:, 1] * 0.5 + X_w[:, 2] * (0.2 if w < 5 else 0.9)
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
    print(f"Generated synthetic drift dataset ({len(df)} rows) -> {output_path}")
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
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset == "fraud":
        download_fraud_dataset(args.output_dir)
    elif args.dataset == "synthetic":
        generate_synthetic_drift_data(os.path.join(args.output_dir, "synthetic_drift.csv"))
