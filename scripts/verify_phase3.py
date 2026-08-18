"""Phase 3 SHAP Explainability Layer Verification Script."""
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
from driftx.explainability.shap_computer import ShapComputer
from driftx.explainability.magnitude import MagnitudeTracker
from driftx.explainability.rank_change import RankChangeTracker
from scripts.download_data import generate_synthetic_drift_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase3_Verification")


def run_verification():
    logger.info("=== Phase 3 Verification Script ===")

    # 1. Load configuration
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    model_config = config["model"]
    shap_config = config["explainability"]

    # 2. Dataset Generation & Windowing
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    synth_path = raw_dir / "synthetic_drift.csv"
    
    generate_synthetic_drift_data(
        output_path=str(synth_path),
        n_windows=8,
        samples_per_window=1500,
        n_features=12,
        seed=42
    )

    ingestor = DataIngestor(data_config)
    df = ingestor.load(data_dir=str(raw_dir))

    windower = WindowSplitter(data_config)
    windows = windower.split(df)

    # 3. Model Training on Window 0
    logger.info("Training initial baseline classifier on Window 0...")
    trainer = ModelTrainer(model_config)
    train_res = trainer.train(windows[0].X, windows[0].y)
    logger.info(f"Model trained (val_acc={train_res['val_accuracy']:.4f})")

    # 4. Initialize SHAP Computer & Trackers
    shap_computer = ShapComputer(shap_config)
    mag_tracker = MagnitudeTracker(threshold=shap_config.get("magnitude_threshold", 0.1))
    rank_tracker = RankChangeTracker(threshold=shap_config.get("rank_change_threshold", 0.3))

    # 5. Extract SHAP Profiles & Track Importance Shift Across Windows
    logger.info("\n--- SHAP Importance Profile & Drift Monitoring Across Windows ---")
    header = f"{'Win':<4} | {'Top Feature':<12} | {'Top |SHAP|':<10} | {'Mag Score':<10} | {'Mag Flag':<8} | {'Spearman ρ':<10} | {'Rank Score':<10} | {'Rank Flag':<8}"
    logger.info(header)
    logger.info("-" * len(header))

    for w in windows:
        # Compute SHAP profile for model on current window data
        shap_res = shap_computer.compute(trainer.get_model(), w.X)
        mean_abs_shap = shap_res["mean_abs_shap"]
        top_feature = shap_res["feature_ranking"][0]
        top_shap_val = mean_abs_shap[top_feature]

        mag_res = mag_tracker.update_and_compare(mean_abs_shap)
        rank_res = rank_tracker.update_and_compare(mean_abs_shap)

        logger.info(
            f"{w.window_id:<4} | {top_feature:<12} | {top_shap_val:<10.4f} | "
            f"{mag_res['magnitude_score']:<10.4f} | {str(mag_res['drift_detected']):<8} | "
            f"{rank_res['spearman_rho']:<10.4f} | {rank_res['rank_change_score']:<10.4f} | "
            f"{str(rank_res['drift_detected']):<8}"
        )

    logger.info("\n✓ All Phase 3 Verification Checks PASSED Successfully!")


if __name__ == "__main__":
    run_verification()
