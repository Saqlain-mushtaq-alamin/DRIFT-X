"""P2: Statistical drift-triggered retraining (no explainability)."""
from typing import Dict, Any
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class DriftOnlyPolicy(RetrainingPolicy):
    """Retrain when statistical drift exceeds threshold (calibrated z-score). No SHAP used."""
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "Drift-Only (Statistical)"
    
    @property
    def policy_id(self) -> str:
        return "p2_drift_only"
    
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
        if "z_score" in stat_drift:
            z_val = float(stat_drift["z_score"])
            should_retrain = bool(z_val >= self.threshold)
            drift_score = z_val
        else:
            should_retrain = bool(stat_drift.get("drift_detected", False))
            drift_score = float(stat_drift.get("drift_score", 0.0))
        
        return PolicyDecision(
            should_retrain=should_retrain,
            reason=f"P2: Statistical drift score={drift_score:.4f} vs threshold={self.threshold:.2f}, "
                   f"retrain={should_retrain}.",
            policy_name=self.name,
            window_id=window_id,
            evidence={"drift_score": drift_score, "threshold": self.threshold}
        )
