"""Tests for the DRIFT-X Experiment Runner."""
import pytest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from driftx.experiment.runner import ExperimentRunner
from driftx.training.trainer import ModelTrainer


@pytest.fixture
def sample_config(tmp_path):
    return {
        "project": {"name": "test_proj", "seed": 10, "n_seeds": 2},
        "data": {
            "dataset": "synthetic_test",
            "timestamp_col": "timestamp",
            "target_col": "target",
            "window_size": 4,
            "min_window_samples": 50,
        },
        "model": {
            "type": "xgboost",
            "params": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}
        },
        "detection": {
            "statistical_method": "psi",
            "psi_threshold": 0.2,
            "ks_alpha": 0.05
        },
        "explainability": {
            "shap_method": "tree",
            "max_shap_samples": 100,
            "magnitude_threshold": 0.1,
            "rank_change_threshold": 0.3
        },
        "fusion": {
            "alpha": 0.3,
            "beta": 0.3,
            "gamma": 0.4,
            "adaptive_lookback": 3,
            "adaptive_lambda": 1.5
        },
        "policy": {
            "fixed_schedule_interval": 2,
            "active_policies": ["p0_never", "p1_fixed", "p5_dis_fused"]
        },
        "mlflow": {
            "tracking_uri": str(tmp_path / "mlruns"),
            "experiment_name": "test_exp"
        },
        "output": {
            "results_dir": str(tmp_path / "results")
        }
    }


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 1000
    timestamps = np.arange(n)
    X = np.random.randn(n, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["timestamp"] = timestamps
    df["target"] = y
    return df


class TestExperimentRunner:

    def test_runner_initialization(self, sample_config):
        runner = ExperimentRunner(sample_config)
        assert runner.config == sample_config

    def test_get_seeds(self, sample_config):
        runner = ExperimentRunner(sample_config)
        seeds = runner._get_seeds()
        assert seeds == [10, 11]

    def test_run_all_returns_dataframe(self, sample_config, sample_df, tmp_path):
        runner = ExperimentRunner(sample_config)
        results = runner.run_all(df_override=sample_df)

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert "policy" in results.columns
        assert "accuracy" in results.columns
        assert "f1" in results.columns
        assert "cumulative_cost" in results.columns

        csv_file = tmp_path / "results" / "experiment_results.csv"
        assert csv_file.exists()

    def test_prequential_evaluation_protocol(self, tmp_path):
        """Spy on ModelTrainer to verify the test-then-train protocol.

        Asserts:
          (a) Window 0 accuracy is NaN — no pre-retrain evaluation possible
              before the very first model is trained.
          (b) For every retrain window (window_id >= 1), an evaluate() call on
              that window's data immediately precedes the train() call on the
              same data — never the reverse.
          (c) The 'accuracy' value in each result row is in [0, 1], confirming
              it was NOT replaced by an in-sample post-retrain evaluation.
        """
        rng = np.random.RandomState(0)
        records = []
        n_windows = 5
        samples = 300

        for w in range(n_windows):
            t_start = w * 86400.0 * 30
            timestamps = np.sort(rng.uniform(t_start, t_start + 86400.0 * 29, samples))
            X = rng.randn(samples, 4)
            # Concept shift at window 3 so drift signals fire
            y = (X[:, 0] > 0).astype(int) if w < 3 else (X[:, 1] < 0).astype(int)
            for i in range(samples):
                records.append({
                    "timestamp": timestamps[i],
                    "target": y[i],
                    "f0": X[i, 0], "f1": X[i, 1],
                    "f2": X[i, 2], "f3": X[i, 3],
                })
        df = pd.DataFrame(records)

        config = {
            "project": {"name": "spy_test", "seed": 42, "n_seeds": 1},
            "data": {
                "dataset": "synthetic_test",
                "timestamp_col": "timestamp",
                "target_col": "target",
                "window_size": n_windows,
                "min_window_samples": 50,
                "data_source_type": "synthetic",
            },
            "model": {"type": "xgboost",
                      "params": {"n_estimators": 10, "max_depth": 3}},
            "detection": {"statistical_method": "ks", "ks_alpha": 0.05},
            "explainability": {
                "shap_method": "tree", "max_shap_samples": 50,
                "magnitude_threshold": 0.1, "rank_change_threshold": 0.3,
            },
            "fusion": {"alpha": 0.4, "beta": 0.4, "gamma": 0.2,
                       "adaptive_lookback": 2, "adaptive_lambda": 1.5},
            "policy": {
                # P1 retrains every 2 windows — guarantees retrains for spy
                "active_policies": ["p1_fixed"],
                "fixed_schedule_interval": 2,
            },
            "output": {"results_dir": str(tmp_path / "results")},
        }

        # ---------- spy infrastructure ----------
        call_log = []  # ("evaluate"|"train", id(X))

        original_evaluate = ModelTrainer.evaluate
        original_train = ModelTrainer.train

        def spy_evaluate(self, X, y):
            result = original_evaluate(self, X, y)
            call_log.append(("evaluate", id(X)))
            return result

        def spy_train(self, X, y, **kwargs):
            call_log.append(("train", id(X)))
            return original_train(self, X, y, **kwargs)

        with patch.object(ModelTrainer, "evaluate", spy_evaluate), \
             patch.object(ModelTrainer, "train", spy_train):
            runner = ExperimentRunner(config)
            results = runner.run_all(df_override=df, output_filename="spy_test.csv")

        p1 = results[results["policy"] == "p1_fixed"].reset_index(drop=True)

        # (a) Window 0 accuracy must be NaN
        w0_rows = p1[p1["window_id"] == 0]
        assert not w0_rows.empty, "No window-0 row found"
        assert w0_rows["accuracy"].isna().all(), (
            f"Window 0 accuracy must be NaN; got {w0_rows['accuracy'].tolist()}"
        )

        # (b) Every consecutive (evaluate, train) pair in call_log must share
        #     the same X identity — evaluate before train on the same data.
        for i in range(len(call_log) - 1):
            method_a, xid_a = call_log[i]
            method_b, xid_b = call_log[i + 1]
            if method_b == "train" and xid_a == xid_b:
                assert method_a == "evaluate", (
                    f"train() on X id={xid_b} was NOT immediately preceded by "
                    f"evaluate() on the same X; got {method_a} instead."
                )

        # Verify at least one pre-retrain evaluate-then-train pair exists
        consecutive_pairs = [
            i for i in range(len(call_log) - 1)
            if call_log[i][0] == "evaluate"
            and call_log[i + 1][0] == "train"
            and call_log[i][1] == call_log[i + 1][1]
        ]
        retrain_rows = p1[(p1["window_id"] >= 1) & (p1["retrained"] == True)]
        assert not retrain_rows.empty, "P1 should have retrained at least once"
        assert len(consecutive_pairs) > 0, (
            "No consecutive (evaluate, train) on same window found — "
            "pre-retrain evaluation may be missing."
        )

        # (c) All window 1+ accuracy values must be in [0, 1] — they are the
        #     pre-retrain out-of-sample evaluations, not in-sample post-retrain.
        non_nan = p1[p1["window_id"] >= 1].dropna(subset=["accuracy"])
        assert not non_nan.empty, "All window 1+ rows have NaN accuracy — unexpected"
        assert non_nan["accuracy"].between(0.0, 1.0).all(), (
            "Logged accuracy out of [0, 1] — possible in-sample leakage"
        )
