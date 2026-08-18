"""Phase 1 Data Ingestion & Windowing Verification Script."""
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.data.ingestor import DataIngestor
from driftx.data.windower import WindowSplitter
from scripts.download_data import generate_synthetic_drift_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase1_Verification")


def run_verification():
    logger.info("=== Phase 1 Verification Script ===")

    # 1. Load config
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    if not config_path.exists():
        logger.error("configs/default.yaml not found")
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    data_config = config["data"]
    logger.info(f"Loaded config: {data_config}")

    # 2. Generate synthetic organic drift dataset for verification
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    synthetic_file = raw_dir / "synthetic_drift.csv"
    
    logger.info("Generating synthetic organic drift dataset...")
    generate_synthetic_drift_data(
        output_path=str(synthetic_file),
        n_windows=8,
        samples_per_window=1500,
        n_features=12,
        seed=42
    )

    # 3. Test DataIngestor
    logger.info("Testing DataIngestor...")
    ingestor = DataIngestor(data_config)
    df = ingestor.load(data_dir=str(raw_dir))
    
    assert len(df) == 12000, f"Expected 12000 rows, got {len(df)}"
    assert df[data_config["timestamp_col"]].is_monotonic_increasing, "DataFrame is not sorted chronologically"
    logger.info("✓ DataIngestor test passed!")

    # 4. Test WindowSplitter
    logger.info("Testing WindowSplitter...")
    windower = WindowSplitter(data_config)
    windows = windower.split(df)

    logger.info(f"Created {len(windows)} windows.")
    assert len(windows) >= 5, f"Expected at least 5 windows, got {len(windows)}"

    for w in windows:
        assert w.n_samples >= data_config.get("min_window_samples", 1000), (
            f"Window {w.window_id} has {w.n_samples} samples (< {data_config['min_window_samples']})"
        )
        assert not w.X.isna().any().any(), f"Window {w.window_id} contains NaNs"
        assert len(w.X) == len(w.y), f"Window {w.window_id} feature and target shape mismatch"

    # 5. Check organic drift presence across windows
    logger.info("\n--- Temporal Window Drift Overview ---")
    for w in windows:
        pos_ratio = w.y.mean()
        f0_mean = w.X["feature_0"].mean()
        f2_mean = w.X["feature_2"].mean()
        logger.info(
            f"Window {w.window_id} (n={w.n_samples}): Target Pos Ratio={pos_ratio:.3f}, "
            f"feat_0 mean={f0_mean:.3f}, feat_2 mean={f2_mean:.3f}"
        )

    logger.info("\n✓ All Phase 1 Verification Checks PASSED Successfully!")


if __name__ == "__main__":
    run_verification()
