"""Drift Impact Score (DIS) — Novel Fused Metric with Gating & Signal Clipping."""
import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DriftImpactScore:
    """
    Computes the Drift Impact Score by fusing three drift signals.

    Supports Gated DIS to prevent a single runaway channel (e.g. continuous covariate
    KS drift) from triggering retraining without explainability consensus:
    
    DIS(t) = α · StatDrift(t) + β · ShapMagnitude(t) + γ · ShapRankChange(t)
    subject to channel clipping (z_max) and gating consensus (min active channels).
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        clip_max: Optional[float] = None,
        gating_mode: str = "none",
        gating_channel_threshold: float = 2.5,
        gating_min_channels: int = 2,
    ):
        """
        Args:
            alpha: Weight for statistical drift signal
            beta: Weight for SHAP magnitude signal
            gamma: Weight for SHAP rank-change signal
            clip_max: Upper clipping bound for z-scores (e.g. 5.0)
            gating_mode: 'two_channel', 'stat_and_shap', 'min', or 'none'
            gating_channel_threshold: Channel z-score threshold to be considered active
            gating_min_channels: Number of active channels required (for 'two_channel')
        """
        total = alpha + beta + gamma
        self.alpha = float(alpha / total)
        self.beta = float(beta / total)
        self.gamma = float(gamma / total)
        self.clip_max = clip_max
        self.gating_mode = gating_mode
        self.gating_channel_threshold = gating_channel_threshold
        self.gating_min_channels = gating_min_channels

        self.history: List[Dict[str, Any]] = []

    def compute(
        self,
        stat_drift_score: float,
        shap_magnitude_score: float,
        shap_rank_change_score: float,
        window_id: int = -1,
        clip_max: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute fused DIS value with clipping and gating.

        Args:
            stat_drift_score: Statistical drift score or calibrated z-score
            shap_magnitude_score: SHAP magnitude change score or calibrated z-score
            shap_rank_change_score: SHAP rank-change score or calibrated z-score
            window_id: Window identifier
            clip_max: Optional override for signal clipping

        Returns:
            Dict containing dis, raw_dis, is_gated, active_channels, components, raw_signals, weights, and window_id.
        """
        effective_clip = clip_max if clip_max is not None else self.clip_max

        s_stat = float(np.clip(stat_drift_score, 0.0, effective_clip) if effective_clip else max(0.0, stat_drift_score))
        s_mag = float(np.clip(shap_magnitude_score, 0.0, effective_clip) if effective_clip else max(0.0, shap_magnitude_score))
        s_rank = float(np.clip(shap_rank_change_score, 0.0, effective_clip) if effective_clip else max(0.0, shap_rank_change_score))

        raw_dis = self.alpha * s_stat + self.beta * s_mag + self.gamma * s_rank

        # Check channel activations
        stat_active = s_stat >= self.gating_channel_threshold
        mag_active = s_mag >= self.gating_channel_threshold
        rank_active = s_rank >= self.gating_channel_threshold
        active_count = int(stat_active) + int(mag_active) + int(rank_active)

        if self.gating_mode == "two_channel":
            is_gated = active_count >= self.gating_min_channels
            dis = raw_dis if is_gated else 0.0
        elif self.gating_mode == "stat_and_shap":
            # Requires statistical drift AND at least one explainability signal
            is_gated = stat_active and (mag_active or rank_active)
            dis = raw_dis if is_gated else 0.0
        elif self.gating_mode == "min":
            # Continuous conservative gate
            is_gated = True
            dis = float(min(s_stat, max(s_mag, s_rank)))
        else:  # "none"
            is_gated = True
            dis = raw_dis

        result = {
            "dis": float(dis),
            "raw_dis": float(raw_dis),
            "is_gated": bool(is_gated),
            "active_channels": active_count,
            "channel_actives": {
                "stat": bool(stat_active),
                "magnitude": bool(mag_active),
                "rank": bool(rank_active),
            },
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
            f"DIS(w{window_id}) = {dis:.4f} (raw={raw_dis:.3f}, gate={is_gated}, active={active_count}/3) "
            f"[stat={self.alpha * s_stat:.3f} + mag={self.beta * s_mag:.3f} + rank={self.gamma * s_rank:.3f}]"
        )
        return result

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

    def get_dis_series(self) -> List[float]:
        return [h["dis"] for h in self.history]
