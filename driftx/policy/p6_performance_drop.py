"""P6: Performance-triggered retraining (retrain when accuracy drops by delta)."""
import numpy as np
from typing import Dict, Any, Optional
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class PerformanceDropPolicy(RetrainingPolicy):
    """
    Retrains when evaluation accuracy drops by delta from post-train baseline.
    Requires labels at decision time (or delayed labels if label_delay is set).
    """

    def __init__(self, delta: float = 0.05):
        self.delta = delta
        self.baseline_accuracy: Optional[float] = None

    @property
    def name(self) -> str:
        return "Performance-Trigger"

    @property
    def policy_id(self) -> str:
        return "p6_performance_drop"

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
        eval_metrics = kwargs.get("eval_metrics", {})
        curr_acc = eval_metrics.get("accuracy", float("nan")) if eval_metrics else float("nan")

        if np.isnan(curr_acc):
            return PolicyDecision(
                should_retrain=False,
                reason="P6: Accuracy unavailable (label delay or warmup).",
                policy_name=self.name,
                window_id=window_id,
            )

        if self.baseline_accuracy is None:
            self.baseline_accuracy = curr_acc
            return PolicyDecision(
                should_retrain=False,
                reason=f"P6: Initialized baseline accuracy to {curr_acc:.4f}.",
                policy_name=self.name,
                window_id=window_id,
                evidence={"baseline_accuracy": self.baseline_accuracy, "current_accuracy": curr_acc},
            )

        drop = self.baseline_accuracy - curr_acc
        should_retrain = bool(drop >= self.delta)
        prev_baseline = self.baseline_accuracy
        if should_retrain:
            self.baseline_accuracy = None  # Re-initialize on newly retrained model's next window

        return PolicyDecision(
            should_retrain=should_retrain,
            reason=f"P6: Accuracy drop={drop:.4f} vs delta={self.delta:.4f} (baseline={prev_baseline:.4f}, curr={curr_acc:.4f})",
            policy_name=self.name,
            window_id=window_id,
            evidence={"drop": drop, "delta": self.delta, "baseline_accuracy": prev_baseline, "current_accuracy": curr_acc},
        )
