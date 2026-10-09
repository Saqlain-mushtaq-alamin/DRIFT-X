"""P9: SHAP of the Loss Retraining Policy (captures concept drift)."""
import numpy as np
from typing import Dict, Any, Optional
from driftx.policy.base import RetrainingPolicy, PolicyDecision


class ShapLossPolicy(RetrainingPolicy):
    """
    Retrains when SHAP values of the loss (TreeExplainer with log_loss and labels)
    indicate that model errors have shifted significantly across features.
    """

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
        self.previous_loss_profile = None

    @property
    def name(self) -> str:
        return "SHAP-Loss (Concept)"

    @property
    def policy_id(self) -> str:
        return "p9_shap_loss"

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
        trainer = kwargs.get("trainer")
        window = kwargs.get("window")
        shap_computer = kwargs.get("shap_computer")

        if trainer is None or window is None or shap_computer is None:
            return PolicyDecision(
                should_retrain=False,
                reason="P9: Model or window data unavailable for loss SHAP.",
                policy_name=self.name,
                window_id=window_id,
            )

        loss_result = shap_computer.compute_loss_shap(trainer.get_model(), window.X, window.y)
        curr_loss_prof = loss_result["mean_abs_shap"]

        if self.previous_loss_profile is None:
            self.previous_loss_profile = curr_loss_prof
            return PolicyDecision(
                should_retrain=False,
                reason="P9: Initialized loss SHAP reference profile.",
                policy_name=self.name,
                window_id=window_id,
            )

        # Compute normalized L1 change in loss attribution
        p1 = self.previous_loss_profile / (self.previous_loss_profile.sum() + 1e-10)
        p2 = curr_loss_prof / (curr_loss_prof.sum() + 1e-10)
        loss_mag_score = float(np.sum(np.abs(p1 - p2)) / 2.0)

        noise_floor = loss_result.get("noise_floor", {})
        mu = noise_floor.get("mu_mag_noise", 0.01)
        sd = noise_floor.get("sd_mag_noise", 0.005)
        z_score = float(max(0.0, (loss_mag_score - mu) / max(sd, 1e-5)))

        should_retrain = bool(z_score >= self.threshold or loss_mag_score > 0.15)
        if should_retrain:
            self.previous_loss_profile = curr_loss_prof

        return PolicyDecision(
            should_retrain=should_retrain,
            reason=f"P9: Loss SHAP z={z_score:.2f} (mag={loss_mag_score:.4f}) vs threshold={self.threshold:.2f}",
            policy_name=self.name,
            window_id=window_id,
            evidence={"z_score": z_score, "loss_mag_score": loss_mag_score},
        )
