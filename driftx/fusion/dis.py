"""Drift Impact Score (DIS) — Novel Fused Metric."""
import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DriftImpactScore:
    """
    Computes the Drift Impact Score by fusing three drift signals.

    DIS(t) = α · StatDrift(t) + β · ShapMagnitude(t) + γ · ShapRankChange(t)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4):
        """
        Args:
            alpha: Weight for statistical drift signal
            beta: Weight for SHAP magnitude signal
            gamma: Weight for SHAP rank-change signal
        """
        total = alpha + beta + gamma
        self.alpha = float(alpha / total)
        self.beta = float(beta / total)
        self.gamma = float(gamma / total)

        self.history: List[Dict[str, Any]] = []

    def compute(
        self,
        stat_drift_score: float,
        shap_magnitude_score: float,
        shap_rank_change_score: float,
        window_id: int = -1,
    ) -> Dict[str, Any]:
        """
        Compute fused DIS value.

        Args:
            stat_drift_score: Statistical drift ratio/score (0 to 1)
            shap_magnitude_score: SHAP magnitude change score (0 to 1)
            shap_rank_change_score: SHAP rank-change score 1 - ρ (0 to 1)
            window_id: Window identifier

        Returns:
            Dict containing dis, components, raw_signals, weights, and window_id.
        """
        s_stat = float(np.clip(stat_drift_score, 0.0, 1.0))
        s_mag = float(np.clip(shap_magnitude_score, 0.0, 1.0))
        s_rank = float(np.clip(shap_rank_change_score, 0.0, 1.0))

        dis = self.alpha * s_stat + self.beta * s_mag + self.gamma * s_rank

        result = {
            "dis": float(dis),
            "components": {
                "stat_contribution": float(self.alpha * s_stat),
                "magnitude_contribution": float(self.beta * s_mag),
                "rank_change_contribution": float(self.gamma * s_rank),
            },
            "raw_signals": {
                "stat_drift": s_stat,
                "shap_magnitude": s_mag,
                "shap_rank_change": s_rank,
            },
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
            "window_id": window_id,
        }

        self.history.append(result)

        logger.info(
            f"DIS(w{window_id}) = {dis:.4f} [stat={self.alpha * s_stat:.3f} + mag={self.beta * s_mag:.3f} + rank={self.gamma * s_rank:.3f}]"
        )
        return result

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

    def get_dis_series(self) -> List[float]:
        return [h["dis"] for h in self.history]
