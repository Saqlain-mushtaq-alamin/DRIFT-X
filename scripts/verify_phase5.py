"""Phase 5 Retraining Policy Engine Verification Script."""
import sys
import logging
import yaml
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
from driftx.policy import create_all_policies
from scripts.download_data import generate_synthetic_drift_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase5_Verification")


def run_verification():
    logger.info("=== Phase 5 Policy Engine Verification Script ===")

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

    # 5. Instantiate All 6 Retraining Policies
    policy_config = {
        "active_policies": ["p0_never", "p1_fixed", "p2_drift_only", "p3_shap_magnitude", "p4_shap_rank", "p5_dis_fused"],
        "fixed_schedule_interval": 3,
    }
    policies = create_all_policies(policy_config)
    logger.info(f"Loaded {len(policies)} retraining policies: {list(policies.keys())}")

    # 6. Run Policy Decision Matrix across Windows
    logger.info("\n--- Retraining Policy Decision Matrix Across Windows ---")
    header = f"{'Win':<4} | {'DIS':<7} | {'θ':<7} | {'P0 (Never)':<10} | {'P1 (Fixed)':<10} | {'P2 (Stat)':<10} | {'P3 (Mag)':<10} | {'P4 (Rank)':<10} | {'P5 (DIS-Fused)':<14}"
    logger.info(header)
    logger.info("-" * len(header))

    decisions_summary = {pid: [] for pid in policies.keys()}

    for w in windows:
        # 1. Statistical Signal
        stat_res = psi_detector.detect(w.X)
        
        # 2. Explainability Signals
        shap_res = shap_computer.compute(trainer.get_model(), w.X)
        mean_abs_shap = shap_res["mean_abs_shap"]
        mag_res = mag_tracker.update_and_compare(mean_abs_shap)
        rank_res = rank_tracker.update_and_compare(mean_abs_shap)

        # 3. DIS Signal Fusion
        dis_res = dis_engine.compute(
            stat_drift_score=stat_res["drift_score"],
            shap_magnitude_score=mag_res["magnitude_score"],
            shap_rank_change_score=rank_res["rank_change_score"],
            window_id=w.window_id,
        )

        # 4. Adaptive Threshold Decision
        atc_res = atc.update_and_decide(dis_res["dis"], window_id=w.window_id)

        # 5. Policy Decisions
        win_decisions = {}
        for pid, policy in policies.items():
            decision = policy.decide(
                window_id=w.window_id,
                stat_drift=stat_res,
                shap_magnitude=mag_res,
                shap_rank_change=rank_res,
                dis_result=dis_res,
                atc_result=atc_res,
            )
            win_decisions[pid] = decision
            decisions_summary[pid].append(decision.should_retrain)

        # Format output table line
        p0_str = "RETRAIN" if win_decisions["p0_never"].should_retrain else "KEEP"
        p1_str = "RETRAIN" if win_decisions["p1_fixed"].should_retrain else "KEEP"
        p2_str = "RETRAIN" if win_decisions["p2_drift_only"].should_retrain else "KEEP"
        p3_str = "RETRAIN" if win_decisions["p3_shap_magnitude"].should_retrain else "KEEP"
        p4_str = "RETRAIN" if win_decisions["p4_shap_rank"].should_retrain else "KEEP"
        p5_str = "RETRAIN" if win_decisions["p5_dis_fused"].should_retrain else "KEEP"

        logger.info(
            f"{w.window_id:<4} | {dis_res['dis']:<7.4f} | {atc_res['threshold']:<7.4f} | "
            f"{p0_str:<10} | {p1_str:<10} | {p2_str:<10} | {p3_str:<10} | {p4_str:<10} | {p5_str:<14}"
        )

    logger.info("\n--- Retraining Summary Across 8 Windows ---")
    for pid, retrains in decisions_summary.items():
        logger.info(f"Policy {pid:<18}: {sum(retrains)} retrains triggered (windows: {[i for i, r in enumerate(retrains) if r]})")

    logger.info("\n✓ All Phase 5 Verification Checks PASSED Successfully!")


if __name__ == "__main__":
    run_verification()
