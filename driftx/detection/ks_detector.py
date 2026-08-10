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

    def detect(self, X_current: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare current window features against reference window using KS test.

        Returns:
            Dict containing drift_detected, drift_score (ratio of drifted features), per_feature p-values/stats.
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

            results[feature] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "drifted": drifted,
            }

        n_tested = len(results)
        drift_score = float(n_drifted / n_tested) if n_tested > 0 else 0.0
        # Flag drift if 50% or more features show statistically significant distribution shift
        drift_detected = bool(drift_score >= 0.5)

        logger.info(
            f"KS-Test results: {n_drifted}/{n_tested} features drifted (score={drift_score:.3f}, detected={drift_detected})"
        )

        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "per_feature": results,
            "n_drifted_features": n_drifted,
            "n_tested_features": n_tested,
        }
