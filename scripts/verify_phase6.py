"""Phase 6 Experiment Runner & MLflow Verification Script."""
import sys
import logging
import yaml
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase6_Verification")


def run_verification():
    logger.info("=== Phase 6 Experiment Runner Verification Script ===")

    # 1. Load configuration
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Set 2 seeds for verification efficiency
    config["project"]["n_seeds"] = 2
    
    # 2. Instantiate and Run ExperimentRunner
    logger.info(f"Instantiating ExperimentRunner for {config['project']['n_seeds']} seeds...")
    runner = ExperimentRunner(config)
    results = runner.run_all()

    # 3. Assert CSV Output Existence and Schema Integrity
    output_dir = Path(config.get("output", {}).get("results_dir", "results"))
    csv_path = output_dir / "experiment_results.csv"
    
    assert csv_path.exists(), f"Output results CSV missing at {csv_path}"
    logger.info(f"✓ Results CSV confirmed at {csv_path}")

    expected_cols = [
        "policy", "policy_name", "seed", "window_id", "accuracy", "f1",
        "retrained", "cost_seconds", "cumulative_cost", "cumulative_retrains",
        "stat_drift_score", "shap_magnitude_score", "shap_rank_change_score",
        "dis", "threshold"
    ]
    for col in expected_cols:
        assert col in results.columns, f"Required column '{col}' missing in experiment results!"

    logger.info(f"✓ Output schema verified: {len(results)} total rows across policies & seeds.")

    # 4. Print Comparative Policy Metrics Summary
    summary = results.groupby("policy_name").agg({
        "accuracy": ["mean", "std"],
        "f1": ["mean", "std"],
        "cumulative_cost": "last",
        "cumulative_retrains": "last",
    })
    logger.info("\n=== Comparative Policy Execution Summary ===")
    logger.info("\n" + str(summary))

    logger.info("\n✓ All Phase 6 Verification Checks PASSED Successfully!")


if __name__ == "__main__":
    run_verification()
