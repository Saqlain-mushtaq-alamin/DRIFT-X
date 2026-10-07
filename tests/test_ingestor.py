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

    # ------------------------------------------------------------------
    # ELEC2 integrity guard tests (Item 4)
    # ------------------------------------------------------------------

    def test_elec2_guard_passes_for_correct_row_count(self, tmp_path):
        """Guard must NOT raise when the CSV has exactly 45,312 rows and
        data_source_type == 'real'."""
        ELEC2_ROWS = 45_312
        df_fake = pd.DataFrame({
            "Timestamp": np.arange(ELEC2_ROWS, dtype=float),
            "target": np.zeros(ELEC2_ROWS, dtype=int),
            "class": ["UP"] * ELEC2_ROWS,
            "nswprice": np.zeros(ELEC2_ROWS),
        })
        csv_path = tmp_path / "electricity.csv"
        df_fake.to_csv(csv_path, index=False)

        config = {
            "dataset": "electricity",
            "timestamp_col": "Timestamp",
            "target_col": "target",
            "data_source_type": "real",   # triggers the guard
        }
        ingestor = DataIngestor(config)
        result = ingestor.load(data_dir=str(tmp_path))
        assert len(result) == ELEC2_ROWS

    def test_elec2_guard_fails_for_wrong_row_count(self, tmp_path):
        """Guard must raise AssertionError when data_source_type == 'real'
        but row count != 45,312.  This catches the silent synthetic-fallback
        scenario where OpenML fell back to the generator."""
        df_fake = pd.DataFrame({
            "Timestamp": np.arange(18_000, dtype=float),
            "target": np.zeros(18_000, dtype=int),
            "class": ["DOWN"] * 18_000,
            "nswprice": np.zeros(18_000),
        })
        csv_path = tmp_path / "electricity.csv"
        df_fake.to_csv(csv_path, index=False)

        config = {
            "dataset": "electricity",
            "timestamp_col": "Timestamp",
            "target_col": "target",
            "data_source_type": "real",   # triggers the guard
        }
        ingestor = DataIngestor(config)
        with pytest.raises(AssertionError, match="ELEC2 integrity check failed"):
            ingestor.load(data_dir=str(tmp_path))

    def test_elec2_guard_skipped_for_synthetic(self, tmp_path):
        """Guard must be SKIPPED (no error) when data_source_type == 'synthetic',
        regardless of row count.  Keeps CI and dev runs working without OpenML."""
        df_fake = pd.DataFrame({
            "Timestamp": np.arange(3_000, dtype=float),
            "target": np.zeros(3_000, dtype=int),
            "class": ["UP"] * 3_000,
            "nswprice": np.zeros(3_000),
        })
        csv_path = tmp_path / "electricity.csv"
        df_fake.to_csv(csv_path, index=False)

        config = {
            "dataset": "electricity",
            "timestamp_col": "Timestamp",
            "target_col": "target",
            "data_source_type": "synthetic",  # guard is skipped
        }
        ingestor = DataIngestor(config)
        result = ingestor.load(data_dir=str(tmp_path))
        assert len(result) == 3_000
