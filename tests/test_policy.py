"""Tests for the retraining policy engine."""
import pytest
from driftx.policy import create_policy, create_all_policies, POLICY_REGISTRY
from driftx.policy.base import PolicyDecision, RetrainingPolicy


# Dummy signal data for testing
DUMMY_SIGNALS_TRUE = {
    "stat_drift": {"drift_detected": True, "drift_score": 0.7},
    "shap_magnitude": {"drift_detected": True, "magnitude_score": 0.5},
    "shap_rank_change": {"drift_detected": True, "rank_change_score": 0.6, "spearman_rho": 0.4},
    "dis_result": {"dis": 0.55, "components": {}},
    "atc_result": {"should_retrain": True, "dis_value": 0.55, "threshold": 0.3, "is_warmup": False},
}

DUMMY_SIGNALS_FALSE = {
    "stat_drift": {"drift_detected": False, "drift_score": 0.05},
    "shap_magnitude": {"drift_detected": False, "magnitude_score": 0.02},
    "shap_rank_change": {"drift_detected": False, "rank_change_score": 0.01, "spearman_rho": 0.99},
    "dis_result": {"dis": 0.03, "components": {}},
    "atc_result": {"should_retrain": False, "dis_value": 0.03, "threshold": 0.15, "is_warmup": True},
}


class TestPolicies:
    
    def test_p0_never_retrains(self):
        p = create_policy("p0_never")
        result = p.decide(window_id=5, **DUMMY_SIGNALS_TRUE)
        assert result.should_retrain is False
        assert result.policy_name == "Never-Retrain"
        assert result.window_id == 5
    
    def test_p1_schedule(self):
        p = create_policy("p1_fixed", interval=3)
        # Window 0 should not trigger
        r0 = p.decide(window_id=0, **DUMMY_SIGNALS_TRUE)
        assert r0.should_retrain is False
        
        # Window 3 should trigger
        r3 = p.decide(window_id=3, **DUMMY_SIGNALS_TRUE)
        assert r3.should_retrain is True
        
        # Window 4 should not trigger
        r4 = p.decide(window_id=4, **DUMMY_SIGNALS_TRUE)
        assert r4.should_retrain is False

        # Window 6 should trigger
        r6 = p.decide(window_id=6, **DUMMY_SIGNALS_TRUE)
        assert r6.should_retrain is True
    
    def test_p2_drift_only(self):
        p = create_policy("p2_drift_only")
        res_true = p.decide(window_id=1, **DUMMY_SIGNALS_TRUE)
        assert res_true.should_retrain is True
        
        res_false = p.decide(window_id=1, **DUMMY_SIGNALS_FALSE)
        assert res_false.should_retrain is False
    
    def test_p3_shap_magnitude(self):
        p = create_policy("p3_shap_magnitude")
        res_true = p.decide(window_id=1, **DUMMY_SIGNALS_TRUE)
        assert res_true.should_retrain is True

        res_false = p.decide(window_id=1, **DUMMY_SIGNALS_FALSE)
        assert res_false.should_retrain is False

    def test_p4_shap_rank(self):
        p = create_policy("p4_shap_rank")
        res_true = p.decide(window_id=1, **DUMMY_SIGNALS_TRUE)
        assert res_true.should_retrain is True

        res_false = p.decide(window_id=1, **DUMMY_SIGNALS_FALSE)
        assert res_false.should_retrain is False

    def test_p5_dis_fused(self):
        p = create_policy("p5_dis_fused")
        res_true = p.decide(window_id=1, **DUMMY_SIGNALS_TRUE)
        assert res_true.should_retrain is True
        assert res_true.policy_name == "DIS-Fused (Novel)"

        res_false = p.decide(window_id=1, **DUMMY_SIGNALS_FALSE)
        assert res_false.should_retrain is False
    
    def test_all_policies_created(self):
        config = {
            "active_policies": [
                "p0_never", "p1_fixed", "p2_drift_only",
                "p3_shap_magnitude", "p4_shap_rank", "p5_dis_fused"
            ],
            "fixed_schedule_interval": 3
        }
        policies = create_all_policies(config)
        assert len(policies) == 6
        for key in config["active_policies"]:
            assert key in policies
            assert isinstance(policies[key], RetrainingPolicy)
    
    def test_all_return_policy_decision(self):
        config = {
            "active_policies": list(POLICY_REGISTRY.keys()),
            "fixed_schedule_interval": 3
        }
        policies = create_all_policies(config)
        for pid, policy in policies.items():
            result = policy.decide(window_id=1, **DUMMY_SIGNALS_TRUE)
            assert isinstance(result, PolicyDecision)
            assert isinstance(result.reason, str)
            assert len(result.reason) > 0
            assert result.window_id == 1

    def test_invalid_policy_id(self):
        with pytest.raises(ValueError, match="Unknown policy: invalid_policy"):
            create_policy("invalid_policy")
