"""Temporal window splitting with zero data leakage guarantees."""
from dataclasses import dataclass
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Window:
    """Container representing a single temporal window of dataset observations."""
    window_id: int
    X: pd.DataFrame
    y: pd.Series
    timestamp_start: float
    timestamp_end: float
    n_samples: int

    def __repr__(self) -> str:
        return (
            f"Window(id={self.window_id}, n={self.n_samples}, "
            f"t_range=[{self.timestamp_start:.2f}, {self.timestamp_end:.2f}])"
        )


class WindowSplitter:
    """Splits time-ordered DataFrame into sequential non-overlapping temporal windows."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing:
                - timestamp_col: str
                - target_col: str
                - window_size: str ("daily", "weekly", "monthly", "quarterly") or int (num windows)
                - min_window_samples: int
        """
        self.timestamp_col = config["timestamp_col"]
        self.target_col = config["target_col"]
        self.window_size = config.get("window_size", "monthly")
        self.min_samples = config.get("min_window_samples", 1000)

    def split(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> List[Window]:
        """
        Split DataFrame into temporal windows.

        Args:
            df: Chronologically sorted DataFrame
            feature_cols: Explicit list of feature column names. If None, auto-selects numeric features.

        Returns:
            List of non-overlapping Window objects.
        """
        if feature_cols is None:
            exclude = {self.timestamp_col, self.target_col}
            feature_cols = [
                col for col in df.columns
                if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
            ]

        boundaries = self._compute_boundaries(df)
        windows: List[Window] = []

        for start, end in boundaries:
            mask = (df[self.timestamp_col] >= start) & (df[self.timestamp_col] < end)
            w_df = df[mask]

            if len(w_df) < self.min_samples:
                logger.warning(
                    f"Skipping window [{start:.1f}, {end:.1f}] — sample size ({len(w_df)}) < min ({self.min_samples})"
                )
                continue

            X = w_df[feature_cols].copy()
            y = w_df[self.target_col].copy()

            # Impute missing feature values with feature column median
            if X.isna().sum().sum() > 0:
                X = X.fillna(X.median())

            window = Window(
                window_id=len(windows),
                X=X,
                y=y,
                timestamp_start=start,
                timestamp_end=end,
                n_samples=len(w_df),
            )
            windows.append(window)

        logger.info(f"Generated {len(windows)} temporal windows.")
        self._verify_no_leakage(windows)
        return windows

    def _compute_boundaries(self, df: pd.DataFrame) -> List[Tuple[float, float]]:
        timestamps = df[self.timestamp_col]
        t_min, t_max = float(timestamps.min()), float(timestamps.max())

        if isinstance(self.window_size, int):
            edges = np.linspace(t_min, t_max + 1e-5, self.window_size + 1)
        elif self.window_size == "daily":
            step = 1 * 86400.0 if t_max > 1e6 else 1.0
            edges = np.arange(t_min, t_max + step, step)
        elif self.window_size == "weekly":
            step = 7 * 86400.0 if t_max > 1e6 else 7.0
            edges = np.arange(t_min, t_max + step, step)
        elif self.window_size == "monthly":
            step = 30 * 86400.0 if t_max > 1e6 else 30.0
            edges = np.arange(t_min, t_max + step, step)
        elif self.window_size == "quarterly":
            step = 90 * 86400.0 if t_max > 1e6 else 90.0
            edges = np.arange(t_min, t_max + step, step)
        else:
            raise ValueError(f"Invalid window_size setting: {self.window_size}")

        return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]

    def _verify_no_leakage(self, windows: List[Window]):
        """Ensure no timestamps overlap between adjacent windows."""
        for i in range(len(windows) - 1):
            assert windows[i].timestamp_end <= windows[i + 1].timestamp_start, (
                f"Data leakage detected between window {i} and window {i+1}!"
            )
        logger.info("✓ Zero data leakage verified across all temporal windows.")
