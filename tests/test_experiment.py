"""Tests for the DRIFT-X Experiment Runner."""
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from driftx.experiment.runner import ExperimentRunner


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
