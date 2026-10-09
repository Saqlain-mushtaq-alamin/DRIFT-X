"""P7: Importance-weighted KS drift detector (weights KS by feature |SHAP|)."""
from typing import Dict, Any
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class ImportanceWeightedKSPolicy(RetrainingPolicy):
    """
    Retrains when importance-weighted KS statistic exceeds threshold.
    Tests whether explanations add anything over cheap statistics by weighting
    each feature's KS statistic by its mean |SHAP|.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "Importance-Weighted KS"

    @property
    def policy_id(self) -> str:
        return "p7_weighted_ks"

    def decide(
        self,
        window_id: int,
        stat_drift: Dict[str, Any],
        shap_magnitude: Dict[str, Any],
        shap_rank_change: Dict[str, Any],
        dis_result: Dict[str, Any],
        atc_result: Dict[str, Any],
        **kwargs: Any,
    ) -> PolicyDecision:
        weighted_d_ratio = stat_drift.get("weighted_d_ratio", stat_drift.get("mean_d_ratio", 0.0))
        should_retrain = bool(weighted_d_ratio >= self.threshold)

        return PolicyDecision(
            should_retrain=should_retrain,
            reason=f"P7: Weighted D/D_crit={weighted_d_ratio:.4f} vs threshold={self.threshold:.4f}",
            policy_name=self.name,
            window_id=window_id,
            evidence={"weighted_d_ratio": weighted_d_ratio, "threshold": self.threshold},
        )
