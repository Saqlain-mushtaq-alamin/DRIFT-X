"""Adaptive Threshold Controller — Dynamic retraining trigger adaptation."""
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveThresholdController:
    """
    Computes a dynamic threshold for retraining decisions based on recent DIS history.

    θ(t) = μ(DIS[t-k:t-1]) + λ · σ(DIS[t-k:t-1])
    """

    def __init__(
        self,
        lookback: int = 10,
        lambda_val: float = 1.92,
        min_threshold: float = 0.05,
        warmup_threshold: float = float("inf"),
        fixed_threshold: Optional[float] = None,
        mode: str = "adaptive",
    ):
        """
        Args:
            lookback: Past window count (k) for moving mean/std
            lambda_val: Sensitivity multiplier (λ), default 1.92 for ~5% false alarms
            min_threshold: Floor for the dynamic threshold
            warmup_threshold: Fallback threshold during warmup (default: inf to never retrain during warmup)
            fixed_threshold: Optional fixed threshold (e.g. 3.0 for calibrated noise units)
            mode: 'adaptive' (dynamic mu + lambda * sigma) or 'fixed' (constant threshold post-warmup)
        """
        self.lookback = lookback
        self.lambda_val = lambda_val
        self.min_threshold = min_threshold
        self.warmup_threshold = warmup_threshold
        self.fixed_threshold = fixed_threshold
        self.mode = mode
        self.dis_history: List[float] = []

    def update_and_decide(
        self,
        dis_value: float,
        window_id: int = -1,
    ) -> Dict[str, Any]:
        """
        Evaluate current DIS against dynamic threshold.

        Returns:
            Dict containing should_retrain, threshold, dis_value, is_warmup, threshold_components.
        """
        threshold, is_warmup, components = self._compute_threshold()
        should_retrain = bool(dis_value > threshold)

        # Always record the DIS value, including retrain-triggering spikes.
        self.dis_history.append(dis_value)

        result = {
            "should_retrain": should_retrain,
            "threshold": float(threshold),
            "dis_value": float(dis_value),
            "is_warmup": is_warmup,
            "threshold_components": components,
            "window_id": window_id,
        }

        logger.info(
            f"ATC(w{window_id}): DIS={dis_value:.4f} vs θ={threshold:.4f} -> "
            f"{'RETRAIN' if should_retrain else 'KEEP'}{' (warmup)' if is_warmup else ''}"
        )
        return result

    def _compute_threshold(self) -> Tuple[float, bool, Dict[str, Any]]:
        warmup_req = 1 if (self.mode == "fixed" and self.fixed_threshold is not None) else self.lookback
        if len(self.dis_history) < warmup_req:
            return self.warmup_threshold, True, {"mean": None, "std": None, "k_used": len(self.dis_history)}

        recent = self.dis_history[-self.lookback:]
        mu = float(np.mean(recent))
        sigma = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0

        if self.mode == "fixed" and self.fixed_threshold is not None:
            threshold = float(self.fixed_threshold)
        else:
            threshold = float(mu + self.lambda_val * sigma)
            threshold = float(max(threshold, self.min_threshold))

        return threshold, False, {"mean": mu, "std": sigma, "k_used": len(recent)}

    def reset(self):
        self.dis_history = []

    def get_threshold_history(self) -> List[float]:
        thresholds = []
        for i in range(len(self.dis_history)):
            if i < self.lookback:
                thresholds.append(self.warmup_threshold)
            else:
                recent = self.dis_history[i - self.lookback : i]
                if self.mode == "fixed" and self.fixed_threshold is not None:
                    thresholds.append(float(self.fixed_threshold))
                else:
                    mu = float(np.mean(recent))
                    sigma = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
                    t = max(mu + self.lambda_val * sigma, self.min_threshold)
                    thresholds.append(float(t))
        return thresholds
