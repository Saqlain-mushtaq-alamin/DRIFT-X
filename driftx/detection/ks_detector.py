"""Kolmogorov-Smirnov (KS-test) feature drift detector."""
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class KSDriftDetector:
    """Detects feature-level statistical distribution drift using the 2-sample Kolmogorov-Smirnov test."""

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level threshold for rejecting null hypothesis of identical distributions.
        """
        self.alpha = alpha
        self.reference_data: Optional[pd.DataFrame] = None

    def set_reference(self, X: pd.DataFrame):
        """Set baseline reference window feature matrix."""
        self.reference_data = X.copy()
        logger.info(f"KS reference established: {X.shape[0]} rows, {X.shape[1]} features.")

    def detect(
        self,
        X_current: pd.DataFrame,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compare current window features against reference window using KS test.
        Calibrated against theoretical noise floor D_crit = 1.358 * sqrt((n+m)/(n*m)).

        Args:
            X_current: Feature DataFrame for current window.
            feature_weights: Optional dictionary mapping feature name to importance weight.

        Returns:
            Dict containing drift_detected, drift_score, mean_d_ratio, z_score, per_feature.
        """
        if self.reference_data is None:
            raise RuntimeError("Reference dataset not set. Call set_reference() first.")

        results = {}
        n_drifted = 0
        features = [col for col in self.reference_data.columns if col in X_current.columns]

        for feature in features:
            ref_vals = self.reference_data[feature].dropna().values
            cur_vals = X_current[feature].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            drifted = bool(p_value < self.alpha)
            if drifted:
                n_drifted += 1

            n = len(ref_vals)
            m = len(cur_vals)
            # Theoretical critical value at alpha=0.05 is 1.358 * sqrt((n+m)/(n*m))
            c_alpha = float(np.sqrt(-0.5 * np.log(self.alpha / 2.0))) if (0.0 < self.alpha < 1.0) else 1.358
            d_crit = float(c_alpha * np.sqrt((n + m) / (n * m)))
            d_ratio = float(stat / (d_crit + 1e-10))

            results[feature] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "drifted": drifted,
                "d_crit": d_crit,
                "d_ratio": d_ratio,
            }

        n_tested = len(results)
        drift_fraction = float(n_drifted / n_tested) if n_tested > 0 else 0.0
        ks_statistics = [v["statistic"] for v in results.values()]
        continuous_score = float(np.mean(ks_statistics)) if ks_statistics else 0.0
        d_ratios = [v["d_ratio"] for v in results.values()]
        mean_d_ratio = float(np.mean(d_ratios)) if d_ratios else 0.0
        avg_d_crit = float(np.mean([v["d_crit"] for v in results.values()])) if results else 0.0

        # Calibrate against null noise floor:
        # Under the null hypothesis of identical distributions, E[D / D_crit] ≈ 0.64
        # and standard error over P features is ≈ 0.1917 / sqrt(P).
        p_count = max(n_tested, 1)
        ks_null_mean = 0.64
        ks_null_std = max(0.1917 / np.sqrt(p_count), 0.02)
        z_score = float(max(0.0, (mean_d_ratio - ks_null_mean) / ks_null_std))

        # Importance-weighted KS if weights provided
        weighted_drift_score = continuous_score
        weighted_d_ratio = mean_d_ratio
        if feature_weights is not None:
            w_list = [max(float(feature_weights.get(f, 0.0)), 0.0) for f in results.keys()]
            w_sum = sum(w_list)
            if w_sum > 1e-10:
                weighted_drift_score = float(sum(w * v["statistic"] for w, v in zip(w_list, results.values())) / w_sum)
                weighted_d_ratio = float(sum(w * v["d_ratio"] for w, v in zip(w_list, results.values())) / w_sum)

        # Flag drift if 50% or more features drifted OR calibrated ratio exceeds 1.0
        drift_detected = bool(drift_fraction >= 0.5 or mean_d_ratio >= 1.0)

        logger.info(
            f"KS-Test results: {n_drifted}/{n_tested} features drifted "
            f"(continuous_score={continuous_score:.3f}, mean_D/D_crit={mean_d_ratio:.3f}, z={z_score:.2f}, detected={drift_detected})"
        )

        return {
            "drift_detected": drift_detected,
            "drift_score": continuous_score,       # Continuous mean KS statistic [0, 1]
            "mean_d_ratio": mean_d_ratio,          # mean D / D_crit (~0.64 under null, >1 under real drift)
            "d_crit": avg_d_crit,                  # Mean critical value
            "z_score": z_score,                    # Calibrated z-score in noise units
            "weighted_drift_score": weighted_drift_score,
            "weighted_d_ratio": weighted_d_ratio,
            "drift_fraction": drift_fraction,      # Legacy discrete fraction
            "per_feature": results,
            "n_drifted_features": n_drifted,
            "n_tested_features": n_tested,
        }
