"""Population Stability Index (PSI) feature drift detector."""
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PSIDriftDetector:
    """Detects distribution drift using Population Stability Index (PSI)."""

    def __init__(self, threshold: float = 0.2, n_bins: int = 10):
        """
        Args:
            threshold: PSI threshold (default 0.2 represents significant distribution shift).
            n_bins: Discretization bin count.
        """
        self.threshold = threshold
        self.n_bins = n_bins
        self.reference_data: Optional[pd.DataFrame] = None

    def set_reference(self, X: pd.DataFrame):
        self.reference_data = X.copy()
        logger.info(f"PSI reference established: {X.shape[0]} rows, {X.shape[1]} features.")

    def _compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Compute PSI = Σ (Current_prop - Ref_prop) * ln(Current_prop / Ref_prop)."""
        combined_min = min(reference.min(), current.min())
        combined_max = max(reference.max(), current.max())

        if combined_min == combined_max:
            return 0.0

        bins = np.linspace(combined_min - 1e-6, combined_max + 1e-6, self.n_bins + 1)
        ref_counts = np.histogram(reference, bins=bins)[0]
        cur_counts = np.histogram(current, bins=bins)[0]

        eps = 1e-8
        ref_props = (ref_counts + eps) / (ref_counts.sum() + eps * self.n_bins)
        cur_props = (cur_counts + eps) / (cur_counts.sum() + eps * self.n_bins)

        psi_val = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return float(psi_val)

    def detect(
        self,
        X_current: pd.DataFrame,
        feature_weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self.reference_data is None:
            raise RuntimeError("Reference dataset not set. Call set_reference() first.")

        results = {}
        psi_values = []
        features = [col for col in self.reference_data.columns if col in X_current.columns]

        for feature in features:
            ref_vals = self.reference_data[feature].dropna().values
            cur_vals = X_current[feature].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            psi = self._compute_psi(ref_vals, cur_vals)
            psi_values.append(psi)
            results[feature] = {
                "psi": psi,
                "drifted": bool(psi >= self.threshold),
            }

        mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
        n_drifted = sum(1 for v in results.values() if v["drifted"])
        drift_detected = bool(mean_psi >= self.threshold)

        weighted_psi = mean_psi
        if feature_weights is not None:
            w_list = [max(float(feature_weights.get(f, 0.0)), 0.0) for f in results.keys()]
            w_sum = sum(w_list)
            if w_sum > 1e-10:
                weighted_psi = float(sum(w * v["psi"] for w, v in zip(w_list, results.values())) / w_sum)

        z_score = float(max(0.0, (mean_psi - 0.02) / 0.03))

        logger.info(
            f"PSI results: mean_psi={mean_psi:.4f}, {n_drifted}/{len(results)} features drifted (detected={drift_detected})"
        )

        return {
            "drift_detected": drift_detected,
            "drift_score": mean_psi,
            "weighted_drift_score": weighted_psi,
            "z_score": z_score,
            "per_feature": results,
            "n_drifted_features": n_drifted,
            "mean_psi": mean_psi,
        }
