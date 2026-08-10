"""Data ingestion module for loading and validating DRIFT-X datasets."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataIngestor:
    """Loads raw CSV/Parquet datasets, validates schemas, and sorts chronologically."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary with required keys:
                - dataset: str ("fraud", "intrusion", "synthetic", "healthcare")
                - timestamp_col: str
                - target_col: str
        """
        self.config = config
        self.dataset_name = config.get("dataset", "synthetic")
        self.timestamp_col = config.get("timestamp_col", "TransactionDT")
        self.target_col = config.get("target_col", "isFraud")

    def load(self, data_dir: str = "data/raw") -> pd.DataFrame:
        """
        Load dataset, validate columns, and sort by timestamp.

        Args:
            data_dir: Path to raw dataset directory.

        Returns:
            pd.DataFrame: Validated and chronologically sorted DataFrame.
        """
        loader_map = {
            "fraud": self._load_fraud,
            "intrusion": self._load_intrusion,
            "synthetic": self._load_synthetic,
            "healthcare": self._load_healthcare,
        }

        loader = loader_map.get(self.dataset_name)
        if loader is None:
            raise ValueError(f"Unsupported dataset: '{self.dataset_name}'")

        df = loader(data_dir)
        df = self._validate(df)

        # Sort strictly by timestamp
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        logger.info(
            f"Successfully loaded '{self.dataset_name}': {len(df)} rows, {df.shape[1]} columns. "
            f"Timestamp range: [{df[self.timestamp_col].min():.1f}, {df[self.timestamp_col].max():.1f}]"
        )
        return df

    def _load_fraud(self, data_dir: str) -> pd.DataFrame:
        path = Path(data_dir) / "train_transaction.csv"
        if not path.exists():
            synth_path = Path(data_dir) / "synthetic_drift.csv"
            if synth_path.exists():
                logger.warning(f"File {path} not found. Loading synthetic dataset from {synth_path}.")
                return pd.read_csv(synth_path)
            raise FileNotFoundError(f"Missing dataset at {path}. Run scripts/download_data.py first.")
        
        df = pd.read_csv(path)
        identity_path = Path(data_dir) / "train_identity.csv"
        if identity_path.exists():
            identity = pd.read_csv(identity_path)
            df = df.merge(identity, on="TransactionID", how="left")
        
        return df

    def _load_synthetic(self, data_dir: str) -> pd.DataFrame:
        path = Path(data_dir) / "synthetic_drift.csv"
        if not path.exists():
            from scripts.download_data import generate_synthetic_drift_data
            generate_synthetic_drift_data(str(path))
        return pd.read_csv(path)

    def _load_intrusion(self, data_dir: str) -> pd.DataFrame:
        path = Path(data_dir) / "cicids2018.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing intrusion dataset at {path}")
        df = pd.read_csv(path)
        if "Label" in df.columns and self.target_col not in df.columns:
            df[self.target_col] = (df["Label"] != "Benign").astype(int)
        return df

    def _load_healthcare(self, data_dir: str) -> pd.DataFrame:
        path = Path(data_dir) / "healthcare.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing healthcare dataset at {path}")
        return pd.read_csv(path)

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns, drop target NaNs, and clean types."""
        if self.timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{self.timestamp_col}' not found in dataset")
        if self.target_col not in df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in dataset")

        n_initial = len(df)
        df = df.dropna(subset=[self.target_col])
        if len(df) < n_initial:
            logger.warning(f"Dropped {n_initial - len(df)} rows with missing target values")

        # Cast target to integer
        df[self.target_col] = df[self.target_col].astype(int)
        return df
