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
            "electricity": self._load_electricity,
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
            # Auto-generate synthetic drift data if not present
            from scripts.download_data import generate_synthetic_drift_data
            return generate_synthetic_drift_data(str(synth_path))
        
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
        candidates = [
            Path(data_dir) / "cic_ids2018.csv",
            Path(data_dir) / "cicids2018.csv",
        ]
        found_path = next((p for p in candidates if p.exists()), None)
        if found_path is None:
            logger.info("CIC-IDS2018 file not found in data_dir. Generating realistic drift benchmark...")
            from scripts.download_data import generate_intrusion_drift_data
            target_path = Path(data_dir) / "cic_ids2018.csv"
            df = generate_intrusion_drift_data(str(target_path))
        else:
            df = pd.read_csv(found_path)

        # Standardize target column
        if self.target_col not in df.columns:
            if "is_attack" in df.columns:
                df[self.target_col] = df["is_attack"]
            elif "Label" in df.columns:
                df[self.target_col] = (df["Label"].astype(str) != "Benign").astype(int)
        elif df[self.target_col].dtype == object:
            df[self.target_col] = (df[self.target_col].astype(str) != "Benign").astype(int)

        return df

    def _load_electricity(self, data_dir: str) -> pd.DataFrame:
        candidates = [
            Path(data_dir) / "electricity.csv",
            Path("data/raw/electricity.csv"),
        ]
        found_path = next((p for p in candidates if p.exists()), None)
        if found_path is not None:
            df = pd.read_csv(found_path)
        else:
            from scripts.download_data import download_electricity_dataset
            df = download_electricity_dataset(data_dir)

        # Map target column
        if self.target_col not in df.columns:
            if "target" in df.columns:
                df[self.target_col] = df["target"]
            elif "class" in df.columns:
                df[self.target_col] = (df["class"].astype(str).str.upper() == "UP").astype(int)
        elif df[self.target_col].dtype == object:
            df[self.target_col] = (df[self.target_col].astype(str).str.upper() == "UP").astype(int)

        return df

    def _load_healthcare(self, data_dir: str) -> pd.DataFrame:
        candidates = [
            Path(data_dir) / "healthcare.csv",
            Path(data_dir) / "mimic_clinical.csv",
        ]
        found_path = next((p for p in candidates if p.exists()), None)
        if found_path is None:
            logger.warning("Healthcare dataset not found. Generating synthetic clinical benchmark...")
            np.random.seed(42)
            n_samples = 6000
            timestamps = np.sort(np.random.uniform(0, 90 * 86400.0 * 4, size=n_samples))
            df = pd.DataFrame({
                self.timestamp_col: timestamps,
                self.target_col: np.random.binomial(1, 0.2, size=n_samples),
                "heart_rate": np.random.normal(75, 12, size=n_samples),
                "blood_pressure": np.random.normal(120, 15, size=n_samples),
                "respiratory_rate": np.random.normal(16, 4, size=n_samples),
                "temperature": np.random.normal(37, 0.8, size=n_samples),
                "age": np.random.randint(18, 90, size=n_samples),
            })
            save_path = Path(data_dir) / "healthcare.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            return df
        return pd.read_csv(found_path)

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns, drop target NaNs, and clean types."""
        if self.timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{self.timestamp_col}' not found in dataset")
        if self.target_col not in df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in dataset")

        # Convert datetime timestamp strings to float seconds if necessary
        if not pd.api.types.is_numeric_dtype(df[self.timestamp_col]):
            try:
                dt_series = pd.to_datetime(df[self.timestamp_col])
                df[self.timestamp_col] = (dt_series.astype("int64") / 1e9).astype(float)
            except Exception:
                # Fallback to sequential numeric index
                df[self.timestamp_col] = np.arange(len(df), dtype=float)

        n_initial = len(df)
        df = df.dropna(subset=[self.target_col])
        if len(df) < n_initial:
            logger.warning(f"Dropped {n_initial - len(df)} rows with missing target values")

        # Cast target to integer
        df[self.target_col] = df[self.target_col].astype(int)
        return df
