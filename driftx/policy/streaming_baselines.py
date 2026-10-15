"""Standard concept drift detector baselines from the River library.

Includes:
- ADWIN (Adaptive Windowing)
- DDM (Drift Detection Method)
- KSWIN (Kolmogorov-Smirnov Windowing)
- Page-Hinkley (Sequential analysis change detector)
"""
from typing import Dict, Any, Optional
import numpy as np
from driftx.policy.base import RetrainingPolicy, PolicyDecision

try:
    from river.drift import ADWIN, KSWIN, PageHinkley
    from river.drift.binary import DDM
except ImportError as e:
    raise ImportError("pip install 'river>=0.21' to use p10-p13") from e


class ADWINPolicy(RetrainingPolicy):
    """ADWIN (Adaptive Windowing) drift detection policy."""

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._init_detector()

    def _init_detector(self):
        self.detector = ADWIN(delta=self.delta)

    @property
    def name(self) -> str:
        return "ADWIN (River)"

    @property
    def policy_id(self) -> str:
        return "p10_adwin"

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
        if trainer is None or window is None or window.y is None:
            return PolicyDecision(False, "Data unavailable", self.name, window_id)

        preds = trainer.predict(window.X)
        errors = (preds != window.y).astype(int)
        
        drift_detected = False
        for e in errors:
            self.detector.update(e)
            if self.detector.drift_detected:
                drift_detected = True

        if drift_detected:
            self._init_detector()

        return PolicyDecision(
            should_retrain=drift_detected,
            reason=f"ADWIN: drift_detected={drift_detected} (delta={self.delta})",
            policy_name=self.name,
            window_id=window_id,
            evidence={"drift_detected": drift_detected, "delta": self.delta},
        )


class DDMPolicy(RetrainingPolicy):
    """DDM (Drift Detection Method) binary error rate drift detector."""

    def __init__(self, warm_start: int = 30, warning_threshold: float = 2.0, drift_threshold: float = 3.0):
        self.warm_start = warm_start
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self._init_detector()

    def _init_detector(self):
        self.detector = DDM(
            warm_start=self.warm_start,
            warning_threshold=self.warning_threshold,
            drift_threshold=self.drift_threshold,
        )

    @property
    def name(self) -> str:
        return "DDM (River)"

    @property
    def policy_id(self) -> str:
        return "p11_ddm"

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
        if trainer is None or window is None or window.y is None:
            return PolicyDecision(False, "Data unavailable", self.name, window_id)

        preds = trainer.predict(window.X)
        errors = (preds != window.y).astype(int)

        drift_detected = False
        for e in errors:
            self.detector.update(e)
            if self.detector.drift_detected:
                drift_detected = True

        if drift_detected:
            self._init_detector()

        return PolicyDecision(
            should_retrain=drift_detected,
            reason=f"DDM: drift_detected={drift_detected}",
            policy_name=self.name,
            window_id=window_id,
            evidence={"drift_detected": drift_detected},
        )


class KSWINPolicy(RetrainingPolicy):
    """KSWIN (Kolmogorov-Smirnov Windowing) drift detector on model predictions."""

    def __init__(self, alpha: float = 0.005, window_size: int = 200, stat_size: int = 30, seed: Optional[int] = None):
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self._init_detector()

    def _init_detector(self):
        self.detector = KSWIN(
            alpha=self.alpha,
            window_size=self.window_size,
            stat_size=self.stat_size,
            seed=self.seed,
        )

    @property
    def name(self) -> str:
        return "KSWIN (River)"

    @property
    def policy_id(self) -> str:
        return "p12_kswin"

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
        if trainer is None or window is None:
            return PolicyDecision(False, "Data unavailable", self.name, window_id)

        if hasattr(trainer, "predict_proba"):
            probs = trainer.predict_proba(window.X)[:, 1]
        else:
            probs = trainer.predict(window.X)

        drift_detected = False
        for p in probs:
            self.detector.update(float(p))
            if self.detector.drift_detected:
                drift_detected = True

        if drift_detected:
            self._init_detector()

        return PolicyDecision(
            should_retrain=drift_detected,
            reason=f"KSWIN: drift_detected={drift_detected} (alpha={self.alpha})",
            policy_name=self.name,
            window_id=window_id,
            evidence={"drift_detected": drift_detected, "alpha": self.alpha},
        )


class PageHinkleyPolicy(RetrainingPolicy):
    """Page-Hinkley change detector on prediction errors."""

    def __init__(self, threshold: float = 50.0, delta: float = 0.005):
        self.threshold = threshold
        self.delta = delta
        self._init_detector()

    def _init_detector(self):
        self.detector = PageHinkley(threshold=self.threshold, delta=self.delta)

    @property
    def name(self) -> str:
        return "Page-Hinkley (River)"

    @property
    def policy_id(self) -> str:
        return "p13_page_hinkley"

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
        if trainer is None or window is None or window.y is None:
            return PolicyDecision(False, "Data unavailable", self.name, window_id)

        preds = trainer.predict(window.X)
        errors = (preds != window.y).astype(int)

        drift_detected = False
        for e in errors:
            self.detector.update(float(e))
            if self.detector.drift_detected:
                drift_detected = True

        if drift_detected:
            self._init_detector()

        return PolicyDecision(
            should_retrain=drift_detected,
            reason=f"Page-Hinkley: drift_detected={drift_detected} (threshold={self.threshold})",
            policy_name=self.name,
            window_id=window_id,
            evidence={"drift_detected": drift_detected, "threshold": self.threshold},
        )
