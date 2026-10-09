"""Abstract base class for retraining policies in DRIFT-X."""
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class PolicyDecision:
    """Result of a policy's retrain/keep decision."""
    should_retrain: bool
    reason: str              # Human-readable explanation
    policy_name: str         # Which policy made this decision
    window_id: int
    evidence: Dict[str, Any] = field(default_factory=dict)  # Raw evidence that led to the decision


class RetrainingPolicy(ABC):
    """
    Abstract base class for all retraining policies.
    
    Each policy receives drift signals and decides whether to retrain.
    The Strategy pattern allows swapping policies cleanly.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable policy name."""
        pass
    
    @property
    @abstractmethod
    def policy_id(self) -> str:
        """Machine-readable policy identifier (p0, p1, ...)."""
        pass
    
    @abstractmethod
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
        """
        Make a retrain/keep decision based on available signals.
        
        Args:
            window_id: Current window index
            stat_drift: Output from KSDriftDetector.detect() or PSIDriftDetector.detect()
            shap_magnitude: Output from MagnitudeTracker.update_and_compare()
            shap_rank_change: Output from RankChangeTracker.update_and_compare()
            dis_result: Output from DriftImpactScore.compute()
            atc_result: Output from AdaptiveThresholdController.update_and_decide()
        
        Returns:
            PolicyDecision with the verdict and reasoning
        """
        pass
