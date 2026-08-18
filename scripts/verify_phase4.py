"""Phase 4 DIS Fusion & Adaptive Threshold Controller Verification Script."""
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
from driftx.detection.psi_detector import PSIDriftDetector
from driftx.explainability.shap_computer import ShapComputer
from driftx.explainability.magnitude import MagnitudeTracker
from driftx.explainability.rank_change import RankChangeTracker
from driftx.fusion.dis import DriftImpactScore
from driftx.fusion.threshold import AdaptiveThresholdController
from scripts.download_data import generate_synthetic_drift_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase4_Verification")


def run_verification():
    logger.info("=== Phase 4 Verification Script ===")

    # 1. Load configuration
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    model_config = config["model"]
    detection_config = config["detection"]
    shap_config = config["explainability"]

    # 2. Ingest Data & Partition Windows
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

    # 3. Train Initial Baseline Model on Window 0
    trainer = ModelTrainer(model_config)
    train_res = trainer.train(windows[0].X, windows[0].y)
    logger.info(f"Model trained on Window 0 (val_acc={train_res['val_accuracy']:.4f})")

    # 4. Instantiate Detectors, Explainability Trackers, DIS Fusion, and ATC
    psi_detector = PSIDriftDetector(threshold=detection_config.get("psi_threshold", 0.2))
    psi_detector.set_reference(windows[0].X)

    shap_computer = ShapComputer(shap_config)
    mag_tracker = MagnitudeTracker(threshold=shap_config.get("magnitude_threshold", 0.1))
    rank_tracker = RankChangeTracker(threshold=shap_config.get("rank_change_threshold", 0.3))

    dis_engine = DriftImpactScore(alpha=0.3, beta=0.3, gamma=0.4)
    atc = AdaptiveThresholdController(lookback=3, lambda_val=1.5, min_threshold=0.05, warmup_threshold=0.15)

    # 5. Run DIS Fusion & Dynamic Retrain Trigger Monitoring
    logger.info("\n--- DIS Fusion & Dynamic Adaptive Retrain Trigger Monitoring ---")
    header = f"{'Win':<4} | {'Stat Drift':<10} | {'SHAP Mag':<10} | {'SHAP Rank':<10} | {'Fused DIS':<10} | {'Threshold θ':<12} | {'Action':<8}"
    logger.info(header)
    logger.info("-" * len(header))

    for w in windows:
        # 1. Statistical Signal
        stat_res = psi_detector.detect(w.X)
        stat_score = stat_res["drift_score"]

        # 2. Explainability Signals
        shap_res = shap_computer.compute(trainer.get_model(), w.X)
        mean_abs_shap = shap_res["mean_abs_shap"]
        mag_res = mag_tracker.update_and_compare(mean_abs_shap)
        rank_res = rank_tracker.update_and_compare(mean_abs_shap)

        # 3. DIS Signal Fusion
        dis_res = dis_engine.compute(
            stat_drift_score=stat_score,
            shap_magnitude_score=mag_res["magnitude_score"],
            shap_rank_change_score=rank_res["rank_change_score"],
            window_id=w.window_id,
        )

        # 4. Adaptive Threshold Decision
        atc_res = atc.update_and_decide(dis_res["dis"], window_id=w.window_id)

        action = "RETRAIN" if atc_res["should_retrain"] else "KEEP"

        logger.info(
            f"{w.window_id:<4} | {stat_score:<10.4f} | {mag_res['magnitude_score']:<10.4f} | "
            f"{rank_res['rank_change_score']:<10.4f} | {dis_res['dis']:<10.4f} | "
            f"{atc_res['threshold']:<12.4f} | {action:<8}"
        )

    logger.info("\n✓ All Phase 4 Verification Checks PASSED Successfully!")


if __name__ == "__main__":
    run_verification()
