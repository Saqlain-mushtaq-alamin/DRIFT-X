"""Tests for DIS fusion and adaptive threshold modules."""
import pytest
import numpy as np
from driftx.fusion.dis import DriftImpactScore
from driftx.fusion.threshold import AdaptiveThresholdController


class TestDriftImpactScore:

    def test_weights_normalize(self):
        dis = DriftImpactScore(alpha=1, beta=2, gamma=3)
        total = dis.alpha + dis.beta + dis.gamma
        assert abs(total - 1.0) < 1e-6

    def test_zero_signals(self):
        dis = DriftImpactScore()
        res = dis.compute(0.0, 0.0, 0.0)
        assert res["dis"] == 0.0

    def test_max_signals(self):
        dis = DriftImpactScore(alpha=0.33, beta=0.33, gamma=0.34)
        res = dis.compute(1.0, 1.0, 1.0)
        assert abs(res["dis"] - 1.0) < 0.01

    def test_only_rank_change(self):
        dis = DriftImpactScore(alpha=0.3, beta=0.3, gamma=0.4)
        res = dis.compute(0.0, 0.0, 1.0)
        assert abs(res["dis"] - dis.gamma) < 0.01

    def test_history_tracking(self):
        dis = DriftImpactScore()
        dis.compute(0.1, 0.2, 0.3, window_id=0)
        dis.compute(0.4, 0.5, 0.6, window_id=1)

        assert len(dis.get_history()) == 2
        assert len(dis.get_dis_series()) == 2


class TestAdaptiveThresholdController:

    def test_warmup_uses_fixed_threshold(self):
        atc = AdaptiveThresholdController(lookback=5, warmup_threshold=0.3)
        res = atc.update_and_decide(0.1)
        assert res["is_warmup"] is True
        assert res["threshold"] == 0.3

    def test_adaptive_after_warmup(self):
        atc = AdaptiveThresholdController(lookback=3, lambda_val=1.0)
        for val in [0.1, 0.15, 0.12, 0.11, 0.14]:
            atc.update_and_decide(val)

        res = atc.update_and_decide(0.5)
        assert res["is_warmup"] is False
        assert res["threshold"] != atc.warmup_threshold

    def test_high_dis_triggers_retrain(self):
        atc = AdaptiveThresholdController(lookback=3, lambda_val=1.0, min_threshold=0.1, warmup_threshold=0.2)
        for val in [0.1, 0.1, 0.1, 0.1, 0.1]:
            atc.update_and_decide(val)

        res = atc.update_and_decide(0.9)
        assert res["should_retrain"] is True

    def test_stable_does_not_trigger(self):
        atc = AdaptiveThresholdController(lookback=3, lambda_val=2.0, min_threshold=0.2, warmup_threshold=0.5)
        for val in [0.1, 0.1, 0.1, 0.1, 0.1]:
            atc.update_and_decide(val)

        res = atc.update_and_decide(0.12)
        assert res["should_retrain"] is False

    def test_reset(self):
        atc = AdaptiveThresholdController()
        atc.update_and_decide(0.5)
        atc.reset()
        assert len(atc.dis_history) == 0
