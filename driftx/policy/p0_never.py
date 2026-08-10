"""P0: Never-Retrain baseline — train once, never update."""
from typing import Dict, Any
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class NeverRetrainPolicy(RetrainingPolicy):
    """Baseline that never retrains. Shows degradation under drift."""
    
    @property
    def name(self) -> str:
        return "Never-Retrain"
    
    @property
    def policy_id(self) -> str:
        return "p0_never"
    
    def decide(
        self,
        window_id: int,
        stat_drift: Dict[str, Any],
        shap_magnitude: Dict[str, Any],
        shap_rank_change: Dict[str, Any],
        dis_result: Dict[str, Any],
        atc_result: Dict[str, Any],
    ) -> PolicyDecision:
        return PolicyDecision(
            should_retrain=False,
            reason="Policy P0: Never retrain regardless of drift.",
            policy_name=self.name,
            window_id=window_id,
            evidence={}
        )
