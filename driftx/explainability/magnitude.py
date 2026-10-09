"""SHAP magnitude drift tracking via normalized L1 distance."""
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MagnitudeTracker:
    """Tracks shifts in absolute feature importance magnitude between consecutive windows."""

    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Drift is flagged if normalized magnitude_score > threshold.
        """
        self.threshold = threshold
        self.previous_profile: Optional[pd.Series] = None

    def update_and_compare(
        self,
        current_profile: pd.Series,
        noise_floor: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compare current mean |SHAP| profile against the reference profile.

        Args:
            current_profile: Mean |SHAP| profile for current window.
            noise_floor: Optional dict containing mu_mag_noise and sd_mag_noise.

        Returns:
            Dict containing magnitude_score, z_score, drift_detected, and per_feature_change.
        """
        if self.previous_profile is None:
            self.previous_profile = current_profile.copy()
            return {
                "magnitude_score": 0.0,
                "z_score": 0.0,
                "drift_detected": False,
                "per_feature_change": pd.Series(0.0, index=current_profile.index),
            }

        common_features = self.previous_profile.index.intersection(current_profile.index)
        prev = self.previous_profile[common_features]
        curr = current_profile[common_features]

        # Normalize to relative importance probabilities (sum to 1.0)
        prev_norm = prev / (prev.sum() + 1e-10)
        curr_norm = curr / (curr.sum() + 1e-10)

        # L1 distance normalized to [0, 1] range (max possible L1 distance = 2.0)
        per_feature_change = (curr_norm - prev_norm).abs()
        magnitude_score_normalized = float(per_feature_change.sum() / 2.0)

        z_score = 0.0
        if noise_floor and "mu_mag_noise" in noise_floor:
            mu = noise_floor["mu_mag_noise"]
            sd = noise_floor.get("sd_mag_noise", 1e-4)
            z_score = float(max(0.0, (magnitude_score_normalized - mu) / max(sd, 1e-6)))

        drift_detected = bool(magnitude_score_normalized > self.threshold or z_score >= 3.0)

        logger.info(
            f"SHAP magnitude score: {magnitude_score_normalized:.4f} (z={z_score:.2f}, threshold={self.threshold}, drift={drift_detected})"
        )

        self.previous_profile = current_profile.copy()

        return {
            "magnitude_score": magnitude_score_normalized,
            "z_score": z_score,
            "drift_detected": drift_detected,
            "per_feature_change": per_feature_change,
        }

    def reset(self):
        self.previous_profile = None
