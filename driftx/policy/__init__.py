"""Policy engine — routes to the active policy."""
from typing import Dict, List, Any
from driftx.policy.base import RetrainingPolicy, PolicyDecision
from driftx.policy.p0_never import NeverRetrainPolicy
from driftx.policy.p1_fixed import FixedSchedulePolicy
from driftx.policy.p2_drift_only import DriftOnlyPolicy
from driftx.policy.p3_shap_magnitude import ShapMagnitudePolicy
from driftx.policy.p4_shap_rank import ShapRankChangePolicy
from driftx.policy.p5_dis_fused import DISFusedPolicy
from driftx.policy.p6_performance_drop import PerformanceDropPolicy
from driftx.policy.p7_weighted_ks import ImportanceWeightedKSPolicy
from driftx.policy.p8_random_budget import RandomBudgetPolicy
from driftx.policy.p9_shap_loss import ShapLossPolicy


POLICY_REGISTRY = {
    "p0_never": NeverRetrainPolicy,
    "p1_fixed": FixedSchedulePolicy,
    "p2_drift_only": DriftOnlyPolicy,
    "p3_shap_magnitude": ShapMagnitudePolicy,
    "p4_shap_rank": ShapRankChangePolicy,
    "p5_dis_fused": DISFusedPolicy,
    "p6_performance_drop": PerformanceDropPolicy,
    "p7_weighted_ks": ImportanceWeightedKSPolicy,
    "p8_random_budget": RandomBudgetPolicy,
    "p9_shap_loss": ShapLossPolicy,
}


def create_policy(policy_id: str, **kwargs) -> RetrainingPolicy:
    """Factory function to create a policy by ID."""
    if policy_id not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy: {policy_id}. "
            f"Available: {list(POLICY_REGISTRY.keys())}"
        )
    return POLICY_REGISTRY[policy_id](**kwargs)


def create_all_policies(config: dict) -> Dict[str, RetrainingPolicy]:
    """Create all active policies from config."""
    policies = {}
    active = config.get("active_policies", list(POLICY_REGISTRY.keys()))
    
    for pid in active:
        kwargs = {}
        if pid == "p1_fixed":
            kwargs["interval"] = config.get("fixed_schedule_interval", 3)
        elif pid == "p6_performance_drop":
            kwargs["delta"] = config.get("performance_drop_delta", 0.05)
        elif pid == "p7_weighted_ks":
            kwargs["threshold"] = config.get("weighted_ks_threshold", 1.0)
        elif pid == "p8_random_budget":
            kwargs["retrain_prob"] = config.get("random_retrain_prob", 0.21)
            kwargs["seed"] = config.get("seed", 42)
        elif pid == "p9_shap_loss":
            kwargs["threshold"] = config.get("shap_loss_threshold", 3.0)
        policies[pid] = create_policy(pid, **kwargs)
    
    return policies


__all__ = [
    "RetrainingPolicy",
    "PolicyDecision",
    "NeverRetrainPolicy",
    "FixedSchedulePolicy",
    "DriftOnlyPolicy",
    "ShapMagnitudePolicy",
    "ShapRankChangePolicy",
    "DISFusedPolicy",
    "PerformanceDropPolicy",
    "ImportanceWeightedKSPolicy",
    "RandomBudgetPolicy",
    "ShapLossPolicy",
    "POLICY_REGISTRY",
    "create_policy",
    "create_all_policies",
]
