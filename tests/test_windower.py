"""Tests for the windowing module."""
import pytest
import pandas as pd
import numpy as np
from driftx.data.windower import WindowSplitter, Window


def create_sample_df(n_rows=5000, n_features=10):
    """Create a sample time-ordered DataFrame."""
    np.random.seed(42)
    data = {
        "timestamp": np.sort(np.random.uniform(0, 1000, n_rows)),
        "target": np.random.randint(0, 2, n_rows),
    }
    for i in range(n_features):
        data[f"feature_{i}"] = np.random.randn(n_rows)
    return pd.DataFrame(data)


class TestWindowSplitter:
    
    def test_basic_splitting(self):
        df = create_sample_df()
        config = {
            "timestamp_col": "timestamp",
            "target_col": "target",
            "window_size": 5,
            "min_window_samples": 100,
        }
        splitter = WindowSplitter(config)
        windows = splitter.split(df)
        
        assert len(windows) > 0
        assert all(isinstance(w, Window) for w in windows)
        assert len(windows) == 5

    def test_no_leakage(self):
        df = create_sample_df()
        config = {
            "timestamp_col": "timestamp",
            "target_col": "target",
            "window_size": 10,
            "min_window_samples": 50,
        }
        splitter = WindowSplitter(config)
        windows = splitter.split(df)
        
        for i in range(len(windows) - 1):
            assert windows[i].timestamp_end <= windows[i+1].timestamp_start

    def test_min_samples_filter(self):
        df = create_sample_df(n_rows=100)
        config = {
            "timestamp_col": "timestamp",
            "target_col": "target",
            "window_size": 50,       # Many windows = few samples each
            "min_window_samples": 10,
        }
        splitter = WindowSplitter(config)
        windows = splitter.split(df)
        
        for w in windows:
            assert w.n_samples >= 10

    def test_missing_values_imputation(self):
        df = create_sample_df(n_rows=1000)
        df.loc[10:20, "feature_0"] = np.nan
        config = {
            "timestamp_col": "timestamp",
            "target_col": "target",
            "window_size": 4,
            "min_window_samples": 50,
        }
        splitter = WindowSplitter(config)
        windows = splitter.split(df)
        
        for w in windows:
            assert w.X.isna().sum().sum() == 0

    def test_invalid_window_size(self):
        df = create_sample_df()
        config = {
            "timestamp_col": "timestamp",
            "target_col": "target",
            "window_size": "invalid_mode",
            "min_window_samples": 10,
        }
        splitter = WindowSplitter(config)
        with pytest.raises(ValueError, match="Invalid window_size setting"):
            splitter.split(df)
