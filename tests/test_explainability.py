"""Tests for the SHAP explainability modules."""
import pytest
import numpy as np
import pandas as pd
from driftx.training.trainer import ModelTrainer
from driftx.explainability.shap_computer import ShapComputer
from driftx.explainability.magnitude import MagnitudeTracker
from driftx.explainability.rank_change import RankChangeTracker


@pytest.fixture
def trained_model_and_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "f0": np.random.randn(300),
        "f1": np.random.randn(300) * 2.0,
        "f2": np.random.randn(300) * 0.1,
    })
    y = (X["f0"] * 1.5 + X["f1"] * 0.8 > 0).astype(int)
    
    trainer = ModelTrainer({"type": "xgboost", "params": {"n_estimators": 30, "max_depth": 3}})
    trainer.train(X, y)
    return trainer.get_model(), X


class TestShapComputer:

    def test_compute_shap_tree(self, trained_model_and_data):
        model, X = trained_model_and_data
        computer = ShapComputer({"shap_method": "tree", "max_shap_samples": 150})
        res = computer.compute(model, X)

        assert "shap_values" in res
        assert "mean_abs_shap" in res
        assert "feature_ranking" in res
        assert len(res["feature_ranking"]) == 3
        assert isinstance(res["mean_abs_shap"], pd.Series)

    def test_invalid_method(self, trained_model_and_data):
        model, X = trained_model_and_data
        computer = ShapComputer({"shap_method": "invalid_method"})
        with pytest.raises(ValueError, match="Unsupported SHAP method"):
            computer.compute(model, X)


class TestMagnitudeTracker:

    def test_no_drift(self):
        tracker = MagnitudeTracker(threshold=0.1)
        profile = pd.Series([0.5, 0.3, 0.2], index=["f0", "f1", "f2"])

        res1 = tracker.update_and_compare(profile)
        assert res1["magnitude_score"] == 0.0
        assert res1["drift_detected"] is False

        res2 = tracker.update_and_compare(profile)
        assert res2["magnitude_score"] < 0.01
        assert res2["drift_detected"] is False

    def test_significant_magnitude_drift(self):
        tracker = MagnitudeTracker(threshold=0.1)
        profile1 = pd.Series([0.9, 0.05, 0.05], index=["f0", "f1", "f2"])
        profile2 = pd.Series([0.05, 0.05, 0.9], index=["f0", "f1", "f2"])

        tracker.update_and_compare(profile1)
        res = tracker.update_and_compare(profile2)

        assert res["magnitude_score"] > 0.1
        assert res["drift_detected"] is True

    def test_reset(self):
        tracker = MagnitudeTracker(threshold=0.1)
        profile = pd.Series([0.5, 0.5], index=["f0", "f1"])
        tracker.update_and_compare(profile)
        tracker.reset()
        assert tracker.previous_profile is None


class TestRankChangeTracker:

    def test_identical_ranking(self):
        tracker = RankChangeTracker(threshold=0.3)
        profile1 = pd.Series([0.5, 0.3, 0.2], index=["f0", "f1", "f2"])
        profile2 = pd.Series([0.8, 0.6, 0.4], index=["f0", "f1", "f2"])

        tracker.update_and_compare(profile1)
        res = tracker.update_and_compare(profile2)

        assert res["spearman_rho"] > 0.99
        assert res["rank_change_score"] < 0.05
        assert res["drift_detected"] is False

    def test_inverted_ranking(self):
        tracker = RankChangeTracker(threshold=0.3)
        profile1 = pd.Series([0.5, 0.4, 0.3, 0.2], index=["f0", "f1", "f2", "f3"])
        profile2 = pd.Series([0.2, 0.3, 0.4, 0.5], index=["f0", "f1", "f2", "f3"])

        tracker.update_and_compare(profile1)
        res = tracker.update_and_compare(profile2)

        assert res["spearman_rho"] < 0.0
        assert res["rank_change_score"] > 1.0
        assert res["drift_detected"] is True
        assert len(res["rank_shifts"]) == 4

    def test_reset(self):
        tracker = RankChangeTracker(threshold=0.3)
        profile = pd.Series([0.5, 0.5], index=["f0", "f1"])
        tracker.update_and_compare(profile)
        tracker.reset()
        assert tracker.previous_profile is None
