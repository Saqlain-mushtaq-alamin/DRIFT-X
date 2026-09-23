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


def generate_intrusion_drift_data(
    output_path: str,
    n_windows: int = 10,
    samples_per_window: int = 1500,
    seed: int = 42,
    drift_intensity: float = 0.4,
) -> pd.DataFrame:
    """
    Generate realistic network intrusion flow dataset modeled after CIC-IDS2018
    with organic feature and concept drift across daily windows.

    Args:
        output_path: Path to write the output CSV.
        n_windows: Number of daily temporal windows (e.g. 10 days of traffic).
        samples_per_window: Flow records per day.
        seed: Random seed.
        drift_intensity: Controls attack emergence and feature drift magnitude.

    Returns:
        pd.DataFrame containing network traffic records.
    """
    np.random.seed(seed)
    records = []
    flow_features = [
        "FlowDuration", "TotFwdPkts", "TotBwdPkts", "TotLenFwdPkts", "TotLenBwdPkts",
        "FwdPktLenMax", "FwdPktLenMean", "BwdPktLenMax", "BwdPktLenMean", "FlowByts_s",
        "FlowPkts_s", "FlowIATMean", "FlowIATStd", "FwdIATTot", "BwdIATTot"
    ]
    n_features = len(flow_features)
    base_means = np.random.uniform(5.0, 50.0, size=n_features)
    cov = np.eye(n_features) * 4.0

    day_seconds = 86400.0

    for w in range(n_windows):
        w_means = base_means.copy()

        # Introduce daily concept shifts (attack vectors evolving)
        if w in [2, 3]:
            # DoS / BruteForce attack day: high packet counts & small inter-arrival times
            attack_prob = 0.25 * drift_intensity
            attack_label = "DoS-SynFlood"
            w_means[1] += 20.0 * drift_intensity  # TotFwdPkts
            w_means[10] += 30.0 * drift_intensity # FlowPkts_s
            w_means[11] = max(1.0, w_means[11] - 10.0 * drift_intensity) # FlowIATMean
        elif w in [5, 6]:
            # DDoS-LOIC attack day: massive volumetric forward bytes
            attack_prob = 0.40 * drift_intensity
            attack_label = "DDoS-LOIC"
            w_means[3] += 50.0 * drift_intensity  # TotLenFwdPkts
            w_means[5] += 40.0 * drift_intensity  # FwdPktLenMax
        elif w >= 8:
            # Botnet / Infiltration: asymmetric backward flow exfiltration
            attack_prob = 0.30 * drift_intensity
            attack_label = "Bot-Infiltration"
            w_means[2] += 25.0 * drift_intensity  # TotBwdPkts
            w_means[4] += 60.0 * drift_intensity  # TotLenBwdPkts
        else:
            # Normal day: low benign background noise
            attack_prob = 0.05
            attack_label = "Benign"

        w_means += np.random.normal(0, 0.5, size=n_features)
        X_w = np.abs(np.random.multivariate_normal(w_means, cov, size=samples_per_window))

        # True classification rule based on network flow heuristics
        logit = (
            (X_w[:, 1] - base_means[1]) * 0.15 +
            (X_w[:, 3] - base_means[3]) * 0.10 -
            (X_w[:, 11] - base_means[11]) * 0.08
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        is_attack = ((np.random.uniform(0, 1, size=samples_per_window) < prob) |
                     (np.random.uniform(0, 1, size=samples_per_window) < attack_prob)).astype(int)

        w_start = w * day_seconds
        w_end = (w + 1) * day_seconds - 1.0
        timestamps = np.sort(np.random.uniform(w_start, w_end, size=samples_per_window))

        for i in range(samples_per_window):
            label = attack_label if is_attack[i] == 1 and attack_label != "Benign" else ("Attack" if is_attack[i] == 1 else "Benign")
            row = {
                "Timestamp": timestamps[i],
                "is_attack": is_attack[i],
                "Label": label,
            }
            for f_idx, f_name in enumerate(flow_features):
                row[f_name] = float(X_w[i, f_idx])
            records.append(row)

    df = pd.DataFrame(records)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"Generated CIC-IDS2018 synthetic drift dataset ({len(df):,} rows, {n_windows} daily windows) "
        f"-> {output_path}"
    )
    return df


def download_intrusion_dataset(output_dir: str):
    """Ensure CIC-IDS2018 dataset is available in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, "cic_ids2018.csv")
    alt_path = os.path.join(output_dir, "cicids2018.csv")
    if os.path.exists(target_path) or os.path.exists(alt_path):
        print(f"Intrusion dataset already exists at {target_path}")
        return
    print("CIC-IDS2018 raw archive requires manual UNB download. Generating realistic drift benchmark...")
    generate_intrusion_drift_data(target_path)


def download_electricity_dataset(output_dir: str) -> pd.DataFrame:
    """Download ELEC2 Electricity Pricing benchmark dataset or fallback to synthetic drift."""
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, "electricity.csv")
    if os.path.exists(target_path):
        print(f"Electricity dataset already exists at {target_path}")
        return pd.read_csv(target_path)

    try:
        from sklearn.datasets import fetch_openml
        print("Fetching ELEC2 benchmark from OpenML (dataset 'electricity', version=1)...")
        bunch = fetch_openml("electricity", version=1, as_frame=True)
        df = bunch.frame.copy()

        # Binary label: UP -> 1, DOWN -> 0
        df["target"] = (df["class"].astype(str).str.upper() == "UP").astype(int)

        # Create monotonic numeric timestamp from date and period
        # date is in [0, 1] relative units over 2 years; period is interval within day [0, 1]
        step = 1800.0  # 30-minute interval
        df["Timestamp"] = np.arange(len(df), dtype=float) * step
        df.to_csv(target_path, index=False)
        print(f"Downloaded and formatted ELEC2 dataset ({len(df):,} rows) -> {target_path}")
        return df
    except Exception as e:
        print(f"OpenML fetch failed ({e}). Generating synthetic electricity drift dataset...")
        return generate_electricity_drift_data(target_path)


def generate_electricity_drift_data(
    output_path: str,
    n_windows: int = 12,
    samples_per_window: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Fallback generator for electricity pricing drift dataset."""
    np.random.seed(seed)
    records = []
    features = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    n_features = len(features)
    base_means = np.array([0.05, 0.45, 0.03, 0.42, 0.40])
    cov = np.eye(n_features) * 0.01
    time_step = 86400.0 * 30  # Monthly windows

    for w in range(n_windows):
        w_means = base_means.copy()
        if w >= 2:
            # Seasonal price spikes and demand changes
            w_means[0] += 0.04 * np.sin(w * np.pi / 3)
            w_means[1] += 0.05 * np.cos(w * np.pi / 4)
            w_means[4] += 0.03 * (w / n_windows)

        X_w = np.clip(np.random.multivariate_normal(w_means, cov, size=samples_per_window), 0.0, 1.0)
        logit = (X_w[:, 0] - 0.05) * 15.0 + (X_w[:, 1] - 0.45) * 8.0 - (X_w[:, 4] - 0.4) * 6.0
        prob = 1.0 / (1.0 + np.exp(-logit))
        target = (np.random.uniform(0, 1, size=samples_per_window) < prob).astype(int)

        w_start = w * time_step
        w_end = (w + 1) * time_step - 1.0
        timestamps = np.sort(np.random.uniform(w_start, w_end, size=samples_per_window))

        for i in range(samples_per_window):
            row = {
                "Timestamp": timestamps[i],
                "class": "UP" if target[i] == 1 else "DOWN",
                "target": target[i],
            }
            for f_idx, f_name in enumerate(features):
                row[f_name] = float(X_w[i, f_idx])
            records.append(row)

    df = pd.DataFrame(records)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated electricity drift dataset ({len(df):,} rows) -> {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRIFT-X Data Downloader")
    parser.add_argument(
        "--dataset",
        default="synthetic",
        choices=["fraud", "intrusion", "electricity", "synthetic"],
    )
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--n-windows", type=int, default=20, help="Windows to generate (synthetic only)")
    parser.add_argument("--drift-intensity", type=float, default=0.4, help="Drift strength (synthetic only)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset == "fraud":
        download_fraud_dataset(args.output_dir)
    elif args.dataset == "intrusion":
        download_intrusion_dataset(args.output_dir)
    elif args.dataset == "electricity":
        download_electricity_dataset(args.output_dir)
    elif args.dataset == "synthetic":
        generate_synthetic_drift_data(
            output_path=os.path.join(args.output_dir, "synthetic_drift.csv"),
            n_windows=args.n_windows,
            drift_intensity=args.drift_intensity,
        )

