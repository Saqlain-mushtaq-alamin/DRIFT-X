"""Unit tests for calibrated drift signals, new baselines, and corrected instruments."""
import pytest
import numpy as np
import pandas as pd
from driftx.detection.ks_detector import KSDriftDetector
from driftx.explainability.shap_computer import ShapComputer
from driftx.explainability.magnitude import MagnitudeTracker
from driftx.explainability.rank_change import RankChangeTracker
from driftx.fusion.dis import DriftImpactScore
from driftx.fusion.threshold import AdaptiveThresholdController
from driftx.policy import create_policy
from driftx.experiment.runner import ExperimentRunner
from scripts.run_known_changepoints import compute_changepoint_metrics


class TestCalibratedInstruments:

    def test_replicate_offset_drops_leading_rows(self):
        config = {
            "project": {"seed": 42, "data_replicate_offset_rows": 100},
            "data": {},
        }
        runner = ExperimentRunner(config)
        df = pd.DataFrame({"TransactionDT": np.arange(1000), "val": np.arange(1000)})

        df_seed42 = runner._make_replicate_df(df, seed=42, timestamp_col="TransactionDT")
        assert len(df_seed42) == 1000
        assert df_seed42["val"].iloc[0] == 0

        df_seed43 = runner._make_replicate_df(df, seed=43, timestamp_col="TransactionDT")
        assert len(df_seed43) == 900
        assert df_seed43["val"].iloc[0] == 100

        df_seed44 = runner._make_replicate_df(df, seed=44, timestamp_col="TransactionDT")
        assert len(df_seed44) == 800
        assert df_seed44["val"].iloc[0] == 200

    def test_changepoint_metric_includes_offset_zero(self):
        # A detector that triggers AT the change point (e.g. window 5)
        # must have delay=0 and NOT be counted as an adaptive false trigger.
        records = [
            {"policy": "test_detector", "window_id": 0, "retrained": True},
            {"policy": "test_detector", "window_id": 1, "retrained": False},
            {"policy": "test_detector", "window_id": 2, "retrained": False},
            {"policy": "test_detector", "window_id": 3, "retrained": False},
            {"policy": "test_detector", "window_id": 4, "retrained": False},
            {"policy": "test_detector", "window_id": 5, "retrained": True},  # CP1 trigger
            {"policy": "test_detector", "window_id": 6, "retrained": False},
            {"policy": "test_detector", "window_id": 7, "retrained": False},
            {"policy": "test_detector", "window_id": 8, "retrained": False},
            {"policy": "test_detector", "window_id": 9, "retrained": False},
            {"policy": "test_detector", "window_id": 10, "retrained": True}, # CP2 trigger
        ]
        results_df = pd.DataFrame(records)
        metrics = compute_changepoint_metrics(
            results=results_df,
            change_points=[5, 10],
            n_total_windows=11,
            atc_warmup_windows=3,
        )
        row = metrics.iloc[0]
        assert row["cp1_detection_delay"] == 0
        assert row["cp2_detection_delay"] == 0
        assert row["mean_detection_delay"] == 0.0
        assert row["missed_changepoints"] == 0
        assert row["adaptive_false_trigger_retrains"] == 0
        assert row["adaptive_false_trigger_rate"] == 0.0


class TestCalibratedSignals:

    def test_ks_calibrated_ratio_and_zscore(self):
        np.random.seed(42)
        ref_df = pd.DataFrame({f"f{i}": np.random.randn(500) for i in range(10)})
        cur_df_null = pd.DataFrame({f"f{i}": np.random.randn(500) for i in range(10)})

        detector = KSDriftDetector(alpha=0.05)
        detector.set_reference(ref_df)
        res_null = detector.detect(cur_df_null)

        # Under null, mean D / D_crit should be around 0.64
        assert 0.4 < res_null["mean_d_ratio"] < 0.95
        assert res_null["z_score"] < 3.0  # Null z-score below alarm threshold

        # Under strong drift
        cur_df_drift = pd.DataFrame({f"f{i}": np.random.randn(500) + 2.0 for i in range(10)})
        res_drift = detector.detect(cur_df_drift)
        assert res_drift["mean_d_ratio"] > 1.0
        assert res_drift["z_score"] > 5.0
        assert res_drift["drift_detected"] is True

    def test_ks_importance_weights(self):
        np.random.seed(42)
        ref_df = pd.DataFrame({
            "f0": np.random.randn(500),
            "f1": np.random.randn(500),
        })
        cur_df = pd.DataFrame({
            "f0": np.random.randn(500) + 3.0,  # Drifted
            "f1": np.random.randn(500),        # Stable
        })
        detector = KSDriftDetector(alpha=0.05)
        detector.set_reference(ref_df)

        # Give 100% weight to f0
        res_weighted = detector.detect(cur_df, feature_weights={"f0": 1.0, "f1": 0.0})
        # Give 100% weight to f1
        res_unimportant = detector.detect(cur_df, feature_weights={"f0": 0.0, "f1": 1.0})

        assert res_weighted["weighted_d_ratio"] > res_unimportant["weighted_d_ratio"]

    def test_shap_noise_floor_and_zscores(self):
        computer = ShapComputer({"shap_method": "tree", "max_shap_samples": 200}, seed=42)
        shap_vals = np.random.randn(200, 10) * 0.02
        shap_vals[:, 0] += 0.5  # Feature 0 dominant

        noise_floor = computer.estimate_noise_floor(shap_vals, n_splits=10, top_k=5)
        assert "mu_mag_noise" in noise_floor
        assert "sd_mag_noise" in noise_floor
        assert noise_floor["sd_mag_noise"] > 0

        # MagnitudeTracker with noise floor
        mag_tracker = MagnitudeTracker(threshold=0.1)
        prof_base = pd.Series(np.mean(np.abs(shap_vals), axis=0), index=[f"f{i}" for i in range(10)])
        mag_tracker.update_and_compare(prof_base, noise_floor=noise_floor)

        # Mild noise perturbation -> z_score near 0
        prof_noisy = prof_base + np.random.randn(10) * 0.0001
        res_mag = mag_tracker.update_and_compare(prof_noisy, noise_floor=noise_floor)
        assert res_mag["z_score"] < 2.0

        # RankChangeTracker on top 5
        rank_tracker = RankChangeTracker(threshold=0.3, top_k=5)
        rank_tracker.update_and_compare(prof_base, noise_floor=noise_floor)
        res_rank = rank_tracker.update_and_compare(prof_noisy, noise_floor=noise_floor)
        assert res_rank["z_score"] < 2.0


class TestTriggerFixes:

    def test_warmup_inf_never_retrains(self):
        atc = AdaptiveThresholdController(lookback=5, warmup_threshold=float("inf"))
        for v in [1.0, 5.0, 10.0]:
            res = atc.update_and_decide(v)
            assert res["is_warmup"] is True
            assert res["should_retrain"] is False

    def test_atc_uses_sample_std_ddof_1(self):
        atc = AdaptiveThresholdController(lookback=3, lambda_val=1.0, warmup_threshold=float("inf"), mode="adaptive")
        atc.update_and_decide(10.0)
        atc.update_and_decide(20.0)
        atc.update_and_decide(30.0)
        res = atc.update_and_decide(5.0)
        # sample std of [10, 20, 30] with ddof=1 is 10.0 (whereas ddof=0 is 8.16)
        assert abs(res["threshold_components"]["std"] - 10.0) < 1e-5


class TestNewBaselines:

    def test_performance_drop_policy(self):
        policy = create_policy("p6_performance_drop", delta=0.05)
        # First window initializes baseline
        r1 = policy.decide(1, {}, {}, {}, {}, {}, eval_metrics={"accuracy": 0.85})
        assert r1.should_retrain is False

        # Small drop < 0.05 -> no retrain
        r2 = policy.decide(2, {}, {}, {}, {}, {}, eval_metrics={"accuracy": 0.82})
        assert r2.should_retrain is False

        # Drop >= 0.05 -> retrain
        r3 = policy.decide(3, {}, {}, {}, {}, {}, eval_metrics={"accuracy": 0.79})
        assert r3.should_retrain is True

    def test_weighted_ks_policy(self):
        policy = create_policy("p7_weighted_ks", threshold=1.0)
        r_false = policy.decide(1, {"weighted_d_ratio": 0.7}, {}, {}, {}, {})
        assert r_false.should_retrain is False

        r_true = policy.decide(2, {"weighted_d_ratio": 1.4}, {}, {}, {}, {})
        assert r_true.should_retrain is True

    def test_random_budget_policy(self):
        p_always = create_policy("p8_random_budget", retrain_prob=1.0)
        assert p_always.decide(1, {}, {}, {}, {}, {}).should_retrain is True

        p_never = create_policy("p8_random_budget", retrain_prob=0.0)
        assert p_never.decide(1, {}, {}, {}, {}, {}).should_retrain is False
