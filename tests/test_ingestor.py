"""Tests for the data ingestion module."""
import pytest
import os
import pandas as pd
import numpy as np
from driftx.data.ingestor import DataIngestor
from scripts.download_data import generate_synthetic_drift_data


@pytest.fixture
def synthetic_data_dir(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    generate_synthetic_drift_data(str(data_dir / "synthetic_drift.csv"), n_windows=4, samples_per_window=300)
    return str(data_dir)


class TestDataIngestor:

    def test_load_synthetic_dataset(self, synthetic_data_dir):
        config = {
            "dataset": "synthetic",
            "timestamp_col": "TransactionDT",
            "target_col": "isFraud",
        }
        ingestor = DataIngestor(config)
        df = ingestor.load(data_dir=synthetic_data_dir)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1200
        assert "TransactionDT" in df.columns
        assert "isFraud" in df.columns
        # Check sorting by timestamp
        assert df["TransactionDT"].is_monotonic_increasing

    def test_missing_timestamp_col(self, synthetic_data_dir):
        config = {
            "dataset": "synthetic",
            "timestamp_col": "non_existent_timestamp",
            "target_col": "isFraud",
        }
        ingestor = DataIngestor(config)
        with pytest.raises(KeyError, match="Timestamp column"):
            ingestor.load(data_dir=synthetic_data_dir)

    def test_missing_target_col(self, synthetic_data_dir):
        config = {
            "dataset": "synthetic",
            "timestamp_col": "TransactionDT",
            "target_col": "non_existent_target",
        }
        ingestor = DataIngestor(config)
        with pytest.raises(KeyError, match="Target column"):
            ingestor.load(data_dir=synthetic_data_dir)

    def test_unsupported_dataset(self):
        config = {
            "dataset": "unknown_domain",
            "timestamp_col": "ts",
            "target_col": "y",
        }
        ingestor = DataIngestor(config)
        with pytest.raises(ValueError, match="Unsupported dataset"):
            ingestor.load()
