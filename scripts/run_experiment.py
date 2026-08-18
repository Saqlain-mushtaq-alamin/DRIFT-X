"""CLI entry point for running DRIFT-X experiments."""
import argparse
import yaml
import logging
from pathlib import Path
import sys

try:
    import mlflow
except ImportError:
    mlflow = None

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("RunExperiment")


def main():
    parser = argparse.ArgumentParser(
        description="Run DRIFT-X experiments"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--policies", nargs="+", default=None,
        help="Specific policies to run (e.g., p0_never p5_dis_fused)"
    )
    parser.add_argument(
        "--seeds", type=int, default=None,
        help="Override number of seeds"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
        
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.policies:
        config["policy"]["active_policies"] = args.policies
    if args.seeds:
        config["project"]["n_seeds"] = args.seeds
    
    # Set up MLflow if available
    if mlflow is not None:
        mlflow_cfg = config.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
        exp_name = mlflow_cfg.get("experiment_name", "driftx_main")
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(exp_name)
        except Exception as e:
            logger.warning(f"MLflow setup warning: {e}")
    
    # Run experiments
    runner = ExperimentRunner(config)
    results = runner.run_all()
    
    print(f"\nExperiment complete! {len(results)} rows logged.")
    print(f"Results saved to {config.get('output', {}).get('results_dir', 'results')}/experiment_results.csv")
    
    # Print summary
    if not results.empty and "policy_name" in results.columns:
        summary = results.groupby("policy_name").agg({
            "accuracy": ["mean", "std"],
            "f1": ["mean", "std"],
            "cumulative_cost": "last",
            "cumulative_retrains": "last",
        })
        print("\n=== Experiment Summary ===")
        print(summary)


if __name__ == "__main__":
    main()
