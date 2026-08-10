"""P4: SHAP rank-change-triggered retraining."""
from typing import Dict, Any
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class ShapRankChangePolicy(RetrainingPolicy):
    """Retrain when feature importance ranking shifts significantly."""
    
    @property
    def name(self) -> str:
        return "SHAP-Rank-Change"
    
    @property
    def policy_id(self) -> str:
        return "p4_shap_rank"
    
    def decide(
        self,
        window_id: int,
        stat_drift: Dict[str, Any],
        shap_magnitude: Dict[str, Any],
        shap_rank_change: Dict[str, Any],
        dis_result: Dict[str, Any],
        atc_result: Dict[str, Any],
    ) -> PolicyDecision:
        drift_detected = shap_rank_change.get("drift_detected", False)
        rank_score = shap_rank_change.get("rank_change_score", 0.0)
        rho = shap_rank_change.get("spearman_rho", 1.0)
        
        return PolicyDecision(
            should_retrain=drift_detected,
            reason=f"P4: Rank-change score={rank_score:.4f}, "
                   f"Spearman ρ={rho:.4f}, detected={drift_detected}.",
            policy_name=self.name,
            window_id=window_id,
            evidence={"rank_change_score": rank_score, "spearman_rho": rho}
        )
