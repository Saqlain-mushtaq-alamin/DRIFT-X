"""P8: Random Retraining Policy at fixed budget."""
import numpy as np
from typing import Dict, Any
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class RandomBudgetPolicy(RetrainingPolicy):
    """
    Randomly retrains with probability matching an expected retrain budget.
    Controls for whether retrain gains come purely from model refreshing.
    """

    def __init__(self, retrain_prob: float = 0.21, seed: int = 42):
        self.retrain_prob = retrain_prob
        self.rng = np.random.RandomState(seed)

    @property
    def name(self) -> str:
        return "Random-Budget"

    @property
    def policy_id(self) -> str:
        return "p8_random_budget"

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
        draw = float(self.rng.uniform(0.0, 1.0))
        should_retrain = bool(draw < self.retrain_prob)

        return PolicyDecision(
            should_retrain=should_retrain,
            reason=f"P8: Random draw={draw:.4f} vs p={self.retrain_prob:.4f}",
            policy_name=self.name,
            window_id=window_id,
            evidence={"draw": draw, "retrain_prob": self.retrain_prob},
        )
