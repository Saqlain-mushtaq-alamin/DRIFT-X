"""
P5: DIS-Fused Policy — THE NOVEL RETRAINING POLICY.

Uses the Drift Impact Score with Adaptive Threshold Controller.
This is the main contribution of the DRIFT-X system.
"""
from typing import Dict, Any
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class DISFusedPolicy(RetrainingPolicy):
    """
    Retrain when the fused Drift Impact Score exceeds
    the adaptive threshold.
    
    This policy combines statistical drift, SHAP magnitude,
    and SHAP rank-change through the DIS formula, and uses
    an adaptive threshold that adjusts to the drift regime.
    
    Novel contributions:
    1. First policy to fuse all three signal types
    2. First to use adaptive (not fixed) thresholding on 
       explainability-fused drift scores
    """
    
    @property
    def name(self) -> str:
        return "DIS-Fused (Novel)"
    
    @property
    def policy_id(self) -> str:
        return "p5_dis_fused"
    
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
        should_retrain = atc_result.get("should_retrain", False)
        dis_value = atc_result.get("dis_value", dis_result.get("dis", 0.0))
        threshold = atc_result.get("threshold", 0.0)
        is_warmup = atc_result.get("is_warmup", True)
        
        return PolicyDecision(
            should_retrain=should_retrain,
            reason=f"P5: DIS={dis_value:.4f} vs θ={threshold:.4f} "
                   f"({'warmup' if is_warmup else 'adaptive'}), "
                   f"decision={'RETRAIN' if should_retrain else 'KEEP'}.",
            policy_name=self.name,
            window_id=window_id,
            evidence={
                "dis": dis_value,
                "threshold": threshold,
                "is_warmup": is_warmup,
                "components": dis_result.get("components", {}),
            }
        )
