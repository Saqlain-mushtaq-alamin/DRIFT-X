"""P3: SHAP magnitude-triggered retraining."""
from typing import Dict, Any, Optional
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class ShapMagnitudePolicy(RetrainingPolicy):
    """Retrain when SHAP magnitude change exceeds threshold."""

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "SHAP-Magnitude"

    @property
    def policy_id(self) -> str:
        return "p3_shap_magnitude"

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
        if self.threshold is not None:
            if "z_score" in shap_magnitude:
                drift_detected = bool(shap_magnitude.get("z_score", 0.0) >= self.threshold)
            else:
                drift_detected = bool(shap_magnitude.get("magnitude_score", 0.0) >= self.threshold)
        else:
            drift_detected = shap_magnitude.get("drift_detected", False)
        mag_score = shap_magnitude.get("magnitude_score", 0.0)

        return PolicyDecision(
            should_retrain=drift_detected,
            reason=f"P3: SHAP magnitude_score={mag_score:.4f}, "
                   f"detected={drift_detected}.",
            policy_name=self.name,
            window_id=window_id,
            evidence={"magnitude_score": mag_score}
        )
