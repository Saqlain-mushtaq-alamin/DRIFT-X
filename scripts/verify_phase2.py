"""Phase 2 Baseline Model & Statistical Drift Detection Verification Script."""
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.data.ingestor import DataIngestor
from driftx.data.windower import WindowSplitter
from driftx.training.trainer import ModelTrainer
from driftx.detection.ks_detector import KSDriftDetector
from driftx.detection.psi_detector import PSIDriftDetector
from scripts.download_data import generate_synthetic_drift_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase2_Verification")


def run_verification():
    logger.info("=== Phase 2 Verification Script ===")

    # 1. Load configuration
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    model_config = config["model"]

    # 2. Data generation and ingestion
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    synth_path = raw_dir / "synthetic_drift.csv"
    
    logger.info("Generating synthetic organic drift dataset...")
    generate_synthetic_drift_data(
        output_path=str(synth_path),
        n_windows=8,
        samples_per_window=1500,
        n_features=12,
        seed=42
    )

    ingestor = DataIngestor(data_config)
    df = ingestor.load(data_dir=str(raw_dir))

    # 3. Temporal Window Splitting
    windower = WindowSplitter(data_config)
    windows = windower.split(df)
    logger.info(f"Split dataset into {len(windows)} temporal windows.")

    # 4. Model Training on Window 0 (Baseline)
    logger.info("Training initial baseline XGBoost model on Window 0...")
    trainer = ModelTrainer(model_config)
    train_res = trainer.train(windows[0].X, windows[0].y)
    
    logger.info(f"Initial model training complete: val_acc={train_res['val_accuracy']:.4f}, cost={train_res['cost_seconds']:.3f}s")
    assert train_res["val_accuracy"] > 0.5

    # 5. Initialize Drift Detectors with Window 0 Reference
    ks_detector = KSDriftDetector(alpha=config["detection"].get("ks_alpha", 0.05))
    ks_detector.set_reference(windows[0].X)

    psi_detector = PSIDriftDetector(threshold=config["detection"].get("psi_threshold", 0.2))
    psi_detector.set_reference(windows[0].X)

    # 6. Evaluate Model Performance & Track Drift Across All Windows
    logger.info("\n--- Downstream Window Evaluation & Drift Monitoring ---")
    header = f"{'Win':<4} | {'Accuracy':<8} | {'F1-Score':<8} | {'KS Drift Score':<14} | {'KS Flag':<7} | {'Mean PSI':<8} | {'PSI Flag':<8}"
    logger.info(header)
    logger.info("-" * len(header))

    for w in windows:
        eval_metrics = trainer.evaluate(w.X, w.y)
        ks_res = ks_detector.detect(w.X)
        psi_res = psi_detector.detect(w.X)

        logger.info(
            f"{w.window_id:<4} | {eval_metrics['accuracy']:<8.4f} | {eval_metrics['f1_weighted']:<8.4f} | "
            f"{ks_res['drift_score']:<14.3f} | {str(ks_res['drift_detected']):<7} | "
            f"{psi_res['mean_psi']:<8.4f} | {str(psi_res['drift_detected']):<8}"
        )

    logger.info("\n✓ All Phase 2 Verification Checks PASSED Successfully!")


if __name__ == "__main__":
    run_verification()
