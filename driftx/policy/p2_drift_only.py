"""P2: Statistical drift-triggered retraining (no explainability)."""
from typing import Dict, Any
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class DriftOnlyPolicy(RetrainingPolicy):
    """Retrain when statistical drift exceeds threshold. No SHAP used."""
    
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
    ) -> PolicyDecision:
        drift_detected = stat_drift.get("drift_detected", False)
        drift_score = stat_drift.get("drift_score", 0.0)
        
        return PolicyDecision(
            should_retrain=drift_detected,
            reason=f"P2: Statistical drift_score={drift_score:.4f}, "
                   f"detected={drift_detected}.",
            policy_name=self.name,
            window_id=window_id,
            evidence={"drift_score": drift_score}
        )
