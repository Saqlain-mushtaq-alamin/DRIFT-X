"""SHAP Rank-Change Drift Tracker — novel structural explainability metric."""
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class RankChangeTracker:
    """Tracks changes in feature importance ranking order between consecutive temporal windows using Spearman rank correlation."""

    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: Drift is flagged if rank_change_score (1 - Spearman ρ) > threshold.
        """
        self.threshold = threshold
        self.previous_profile: Optional[pd.Series] = None

    def update_and_compare(self, current_profile: pd.Series) -> Dict[str, Any]:
        """
        Compare current feature importance ranking with previous window's ranking.

        Returns:
            Dict containing rank_change_score (1 - ρ), spearman_rho, spearman_p_value, drift_detected, rank_shifts.
        """
        if self.previous_profile is None:
            self.previous_profile = current_profile.copy()
            return {
                "rank_change_score": 0.0,
                "spearman_rho": 1.0,
                "spearman_p_value": 0.0,
                "drift_detected": False,
                "rank_shifts": pd.DataFrame(),
            }

        common_features = self.previous_profile.index.intersection(current_profile.index)
        prev = self.previous_profile[common_features]
        curr = current_profile[common_features]

        rho, p_value = spearmanr(prev.values, curr.values)
        if np.isnan(rho):
            rho = 1.0
            p_value = 1.0

        # Rank change metric: 1 - ρ (0 = identical ranking, 2 = inverted ranking)
        rank_change_score = float(1.0 - rho)
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
            f"SHAP rank-change: Spearman ρ={rho:.4f}, score={rank_change_score:.4f} (threshold={self.threshold}, drift={drift_detected})"
        )

        self.previous_profile = current_profile.copy()

        return {
            "rank_change_score": rank_change_score,
            "spearman_rho": float(rho),
            "spearman_p_value": float(p_value),
            "drift_detected": drift_detected,
            "rank_shifts": rank_shifts,
        }

    def reset(self):
        self.previous_profile = None
