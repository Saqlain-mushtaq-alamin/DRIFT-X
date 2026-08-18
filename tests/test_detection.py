"""Tests for statistical drift detectors (KS and PSI)."""
import pytest
import numpy as np
import pandas as pd
from driftx.detection.ks_detector import KSDriftDetector
from driftx.detection.psi_detector import PSIDriftDetector


@pytest.fixture
def sample_feature_dfs():
    np.random.seed(42)
    ref_data = {
        f"feat_{i}": np.random.normal(0, 1, 1000) for i in range(5)
    }
    # Current data with drift on feat_0, feat_1, feat_2
    cur_data = {
        "feat_0": np.random.normal(2.5, 1, 1000),  # Drifting
        "feat_1": np.random.normal(-2.0, 1, 1000), # Drifting
        "feat_2": np.random.normal(1.5, 1.5, 1000),# Drifting
        "feat_3": np.random.normal(0, 1, 1000),   # Stable
        "feat_4": np.random.normal(0, 1, 1000),   # Stable
    }
    return pd.DataFrame(ref_data), pd.DataFrame(cur_data)


class TestKSDriftDetector:

    def test_ks_no_reference_raises(self, sample_feature_dfs):
        _, cur_df = sample_feature_dfs
        detector = KSDriftDetector()
        with pytest.raises(RuntimeError, match="Reference dataset not set"):
            detector.detect(cur_df)

    def test_ks_drift_detection(self, sample_feature_dfs):
        ref_df, cur_df = sample_feature_dfs
        detector = KSDriftDetector(alpha=0.05)
        detector.set_reference(ref_df)
        res = detector.detect(cur_df)

        assert res["drift_detected"] is True
        assert res["n_drifted_features"] >= 3
        assert "feat_0" in res["per_feature"]
        assert res["per_feature"]["feat_0"]["drifted"] is True
        assert res["per_feature"]["feat_3"]["drifted"] is False

    def test_ks_no_drift(self, sample_feature_dfs):
        ref_df, _ = sample_feature_dfs
        # Same distribution
        same_df = pd.DataFrame({f"feat_{i}": np.random.normal(0, 1, 1000) for i in range(5)})
        detector = KSDriftDetector(alpha=0.01)
        detector.set_reference(ref_df)
        res = detector.detect(same_df)

        assert res["drift_detected"] is False
        assert res["n_drifted_features"] <= 1


class TestPSIDriftDetector:

    def test_psi_no_reference_raises(self, sample_feature_dfs):
        _, cur_df = sample_feature_dfs
        detector = PSIDriftDetector()
        with pytest.raises(RuntimeError, match="Reference dataset not set"):
            detector.detect(cur_df)

    def test_psi_drift_detection(self, sample_feature_dfs):
        ref_df, cur_df = sample_feature_dfs
        detector = PSIDriftDetector(threshold=0.2, n_bins=10)
        detector.set_reference(ref_df)
        res = detector.detect(cur_df)

        assert res["drift_detected"] is True
        assert res["mean_psi"] > 0.2
        assert res["per_feature"]["feat_0"]["drifted"] is True
        assert res["per_feature"]["feat_3"]["psi"] < 0.2
