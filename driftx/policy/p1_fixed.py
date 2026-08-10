"""P1: Fixed-schedule retraining — retrain every N windows."""
from typing import Dict, Any
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class FixedSchedulePolicy(RetrainingPolicy):
    """Retrain at fixed intervals regardless of drift signals."""
    
    def __init__(self, interval: int = 3):
        """
        Args:
            interval: Retrain every N windows
        """
        self.interval = interval
    
    @property
    def name(self) -> str:
        return f"Fixed-Schedule (every {self.interval})"
    
    @property
    def policy_id(self) -> str:
        return "p1_fixed"
    
    def decide(
        self,
        window_id: int,
        stat_drift: Dict[str, Any],
        shap_magnitude: Dict[str, Any],
        shap_rank_change: Dict[str, Any],
        dis_result: Dict[str, Any],
        atc_result: Dict[str, Any],
    ) -> PolicyDecision:
        should_retrain = (window_id > 0) and (window_id % self.interval == 0)
        return PolicyDecision(
            should_retrain=should_retrain,
            reason=f"P1: Window {window_id} {'is' if should_retrain else 'is not'} "
                   f"on schedule (every {self.interval}).",
            policy_name=self.name,
            window_id=window_id,
            evidence={"interval": self.interval}
        )
