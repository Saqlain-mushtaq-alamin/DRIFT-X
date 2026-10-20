"""Random multi-stream generator for rigorous drift evaluation.

Generates streams with:
- Random change point locations (not aligned with any fixed schedule).
- Harmful concept shifts at change points (centered so Never-Retrain is not pushed below chance).
- Harmless covariate shifts on non-informative features across every window.
- Configurable drift modes: 'mixed', 'gradual_only', 'concept_only', 'abrupt_only'.
"""
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd


def generate_random_drift_stream(
    seed: int,
    n_windows: int = 24,
    n: int = 1500,
    p: int = 15,
    drift_mode: str = "mixed",
    return_info: bool = False,
):
    """
    Generate a non-stationary stream with controlled harmful and harmless drift.

    Args:
        seed: Random seed for reproducibility.
        n_windows: Number of sequential temporal windows.
        n: Number of samples per window.
        p: Total number of features.
        drift_mode: One of 'mixed', 'gradual_only', 'concept_only', 'abrupt_only', 'harmless_only'.
        return_info: If True, returns (df, cps, cp_info) where cp_info maps cp -> is_covariate_visible.

    Returns:
        (df, cps) or (df, cps, cp_info)
    """
    rng = np.random.RandomState(seed)

    # Random change points (not aligned with any fixed retraining schedule)
    n_cp = rng.randint(2, 5)
    cps_all = sorted(rng.choice(range(4, n_windows - 2), n_cp, replace=False))
    cps = [int(c) for c in cps_all] if drift_mode in ("mixed", "concept_only", "abrupt_only") else []
    cp_info = {}

    # Initial informative features and concept coefficients
    informative = rng.choice(p, 4, replace=False)
    coef = np.zeros(p)
    coef[informative] = rng.uniform(0.8, 2.0, 4) * rng.choice([-1, 1], 4)
    mean = np.zeros(p)

    inform_by_window = {}
    rows = []
    for w in range(n_windows):
        # Harmful concept change at a change point: new coefficients
        if w in cps and drift_mode in ("mixed", "concept_only", "abrupt_only"):
            informative = rng.choice(p, 4, replace=False)
            coef = np.zeros(p)
            coef[informative] = rng.uniform(0.8, 2.0, 4) * rng.choice([-1, 1], 4)
            # Sometimes abrupt covariate shift on informative features (visible to KS/SHAP on X)
            cov_visible = False
            if drift_mode != "concept_only" and rng.rand() < 0.5:
                mean[informative] += rng.uniform(-1.5, 1.5, 4)
                cov_visible = True
            cp_info[w] = cov_visible
        
        inform_by_window[w] = informative.copy()
        
        # Harmless drift: slow mean shift on non-informative features every window
        if drift_mode in ("mixed", "gradual_only", "harmless_only"):
            harmless = np.setdiff1d(np.arange(p), informative)
            mean[harmless] += rng.normal(0, 0.15, len(harmless))
            
        if drift_mode == "gradual_only":
            # For pure gradual drift: slow concept shift
            coef += rng.normal(0, 0.05, p)
            
        X = rng.normal(mean, 1.0, size=(n, p))
        # Centering by (X - mean) @ coef preserves balanced decision boundaries
        logits = (X - mean) @ coef
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n) < probs).astype(int)
        
        # Monthly timestamp spacing (approx 30 days = 2.592e6 seconds)
        t = np.sort(rng.uniform(w * 2.592e6, (w + 1) * 2.592e6 - 1, n))
        
        d = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(p)])
        d["TransactionDT"] = t
        d["isFraud"] = y
        d["true_cp"] = int(w in cps)
        rows.append(d)
        
    df = pd.concat(rows, ignore_index=True)
    if return_info:
        return df, cps, cp_info
    return df, cps


if __name__ == "__main__":
    df_test, cps_test = generate_random_drift_stream(seed=1000)
    print(f"Generated stream with {len(df_test)} rows across {df_test['true_cp'].sum()} change points at windows {cps_test}.")
    print(f"Fraud rate: {df_test['isFraud'].mean():.4f}")
