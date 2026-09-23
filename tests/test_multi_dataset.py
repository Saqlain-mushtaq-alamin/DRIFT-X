"""Unit tests for multi-dataset ingestion, daily windowing, and cross-dataset reporting."""
import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from driftx.data.ingestor import DataIngestor
from driftx.data.windower import WindowSplitter
from driftx.experiment.runner import ExperimentRunner
from scripts.generate_tables import generate_cross_dataset_table


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


def test_intrusion_ingestion_and_fallback(tmp_dir):
    config = {
        "dataset": "intrusion",
        "timestamp_col": "Timestamp",
        "target_col": "is_attack",
    }
    ingestor = DataIngestor(config)
    df = ingestor.load(data_dir=tmp_dir)

    assert not df.empty
    assert "Timestamp" in df.columns
    assert "is_attack" in df.columns
    assert set(df["is_attack"].unique()).issubset({0, 1})
    # Check sorted monotonically
    assert (df["Timestamp"].diff().dropna() >= 0).all()


def test_electricity_ingestion_and_fallback(tmp_dir):
    config = {
        "dataset": "electricity",
        "timestamp_col": "Timestamp",
        "target_col": "target",
    }
    ingestor = DataIngestor(config)
    df = ingestor.load(data_dir=tmp_dir)

    assert not df.empty
    assert "Timestamp" in df.columns
    assert "target" in df.columns
    assert set(df["target"].unique()).issubset({0, 1})
    assert (df["Timestamp"].diff().dropna() >= 0).all()


def test_daily_window_splitter(tmp_dir):
    # Create daily timestamp sequence
    n_days = 5
    records = []
    for d in range(n_days):
        t_base = d * 86400.0
        for i in range(200):
            records.append({
                "Timestamp": t_base + i * 400.0,
                "target": i % 2,
                "feature_a": np.random.randn(),
                "feature_b": np.random.randn(),
            })
    df = pd.DataFrame(records)

    config = {
        "timestamp_col": "Timestamp",
        "target_col": "target",
        "window_size": "daily",
        "min_window_samples": 50,
    }
    splitter = WindowSplitter(config)
    windows = splitter.split(df)

    assert len(windows) == n_days
    for i, w in enumerate(windows):
        assert w.window_id == i
        assert len(w.X) == 200
        assert "feature_a" in w.X.columns
        assert "feature_b" in w.X.columns


def test_cross_dataset_table_generation(tmp_path):
    # Create mock result dataframes for 3 datasets
    def make_mock_results(policy_accs):
        rows = []
        for policy, acc in policy_accs.items():
            for seed in [42, 43]:
                for w in range(1, 4):
                    rows.append({
                        "policy": policy,
                        "seed": seed,
                        "window_id": w,
                        "accuracy": acc + np.random.uniform(-0.01, 0.01),
                        "cumulative_cost": 2.5 if policy == "p5_dis_fused" else 1.0,
                        "cumulative_retrains": 2 if policy == "p5_dis_fused" else 1,
                    })
        return pd.DataFrame(rows)

    mock_map = {
        "Fraud": make_mock_results({"p0_never": 0.82, "p2_drift_only": 0.86, "p5_dis_fused": 0.91}),
        "Intrusion": make_mock_results({"p0_never": 0.78, "p2_drift_only": 0.83, "p5_dis_fused": 0.89}),
        "Electricity": make_mock_results({"p0_never": 0.71, "p2_drift_only": 0.75, "p5_dis_fused": 0.81}),
    }

    table_df = generate_cross_dataset_table(dataset_results_map=mock_map, output_dir=str(tmp_path))

    assert not table_df.empty
    assert list(table_df.columns) == ["Dataset", "P0 Acc", "P2 Acc", "P5 Acc", "P5 Retrains", "P5 Cost"]
    assert len(table_df) == 3
    assert (tmp_path / "cross_dataset_table.tex").exists()
    assert (tmp_path / "cross_dataset_table.md").exists()


def test_mini_experiment_on_intrusion_data(tmp_path):
    from scripts.download_data import generate_intrusion_drift_data

    # Generate 3 small daily windows
    csv_file = tmp_path / "cic_ids2018.csv"
    generate_intrusion_drift_data(str(csv_file), n_windows=3, samples_per_window=100)

    config = {
        "project": {"name": "test_exp", "seed": 42, "n_seeds": 1},
        "data": {
            "dataset": "intrusion",
            "timestamp_col": "Timestamp",
            "target_col": "is_attack",
            "window_size": "daily",
            "min_window_samples": 50,
            "data_dir": str(tmp_path),
        },
        "model": {"type": "xgboost", "params": {"n_estimators": 5, "max_depth": 2}},
        "detection": {"statistical_method": "ks", "ks_alpha": 0.05},
        "explainability": {"shap_method": "tree", "max_shap_samples": 50},
        "fusion": {"alpha": 0.4, "beta": 0.4, "gamma": 0.2, "adaptive_lookback": 2, "adaptive_lambda": 1.5},
        "policy": {"active_policies": ["p0_never", "p5_dis_fused"], "fixed_schedule_interval": 2},
        "evaluation": {"metrics": ["accuracy"], "cost_metric": "wall_clock_seconds"},
        "output": {"results_dir": str(tmp_path)},
    }

    runner = ExperimentRunner(config)
    results = runner.run_all(output_filename="test_intrusion_results.csv")

    assert not results.empty
    assert (tmp_path / "test_intrusion_results.csv").exists()
    assert (tmp_path / "latest_model.joblib").exists()
    assert (tmp_path / "latest_dis.json").exists()
