"""SHAP Rank-Change Drift Tracker — novel structural explainability metric."""
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class RankChangeTracker:
    """Tracks changes in feature importance ranking order between consecutive temporal windows."""

    def __init__(self, threshold: float = 0.3, top_k: int = 5, use_weightedtau: bool = False):
        """
        Args:
            threshold: Drift is flagged if rank_change_score > threshold.
            top_k: Number of top features to evaluate rank correlation on (default: 5).
            use_weightedtau: Whether to use scipy.stats.weightedtau instead of Spearman rho.
        """
        self.threshold = threshold
        self.top_k = top_k
        self.use_weightedtau = use_weightedtau
        self.previous_profile: Optional[pd.Series] = None

    def update_and_compare(
        self,
        current_profile: pd.Series,
        noise_floor: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compare current feature importance ranking with previous window's ranking.

        Args:
            current_profile: Mean |SHAP| profile for current window.
            noise_floor: Optional dict containing mu_rank_noise and sd_rank_noise.

        Returns:
            Dict containing rank_change_score, z_score, spearman_rho, drift_detected, rank_shifts.
        """
        if self.previous_profile is None:
            self.previous_profile = current_profile.copy()
            return {
                "rank_change_score": 0.0,
                "z_score": 0.0,
                "spearman_rho": 1.0,
                "spearman_p_value": 0.0,
                "drift_detected": False,
                "rank_shifts": pd.DataFrame(),
            }

        common_features = self.previous_profile.index.intersection(current_profile.index)
        prev = self.previous_profile[common_features]
        curr = current_profile[common_features]

        if self.use_weightedtau:
            from scipy.stats import weightedtau
            tau, p_val = weightedtau(prev.values, curr.values)
            rho = 1.0 if np.isnan(tau) else float(tau)
            p_value = 1.0 if np.isnan(p_val) else float(p_val)
            rank_change_score = float(1.0 - rho)
        elif self.top_k is not None and self.top_k > 0 and len(common_features) > self.top_k:
            # Rank score on top K most important features to eliminate noise-feature rank shuffling
            top_features = prev.nlargest(min(self.top_k, len(prev))).index
            prev_top = prev[top_features]
            curr_top = curr[top_features]
            rho, p_value = spearmanr(prev_top.values, curr_top.values)
            if np.isnan(rho):
                rho = 1.0
                p_value = 1.0
            rank_change_score = float(1.0 - rho)
        else:
            rho, p_value = spearmanr(prev.values, curr.values)
            if np.isnan(rho):
                rho = 1.0
                p_value = 1.0
            rank_change_score = float(1.0 - rho)

        z_score = 0.0
        if noise_floor and "mu_rank_noise" in noise_floor:
            mu = noise_floor["mu_rank_noise"]
            sd = noise_floor.get("sd_rank_noise", 0.05)
            z_score = float(max(0.0, (rank_change_score - mu) / max(sd, 1e-4)))
            drift_detected = bool(z_score >= 3.0)
        else:
            drift_detected = bool(rank_change_score > self.threshold)

        prev_ranks = prev.rank(ascending=False).astype(int)
        curr_ranks = curr.rank(ascending=False).astype(int)

        rank_shifts = pd.DataFrame({
            "prev_rank": prev_ranks,
            "curr_rank": curr_ranks,
            "rank_change": curr_ranks - prev_ranks,
            "abs_rank_change": (curr_ranks - prev_ranks).abs(),
            "prev_shap": prev,
            "curr_shap": curr,
        }).sort_values("abs_rank_change", ascending=False)

        logger.info(
            f"SHAP rank-change: ρ={rho:.4f}, score={rank_change_score:.4f} (z={z_score:.2f}, threshold={self.threshold}, drift={drift_detected})"
        )

        self.previous_profile = current_profile.copy()

        return {
            "rank_change_score": rank_change_score,
            "z_score": z_score,
            "spearman_rho": float(rho),
            "spearman_p_value": float(p_value),
            "drift_detected": drift_detected,
            "rank_shifts": rank_shifts,
        }

    def reset(self):
        self.previous_profile = None
