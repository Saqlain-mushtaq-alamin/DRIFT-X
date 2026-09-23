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
    parser.add_argument(
        "--output", type=str, default=None,
        help="Custom output CSV filename or path"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
        
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        loaded_config = yaml.safe_load(f)

    # If loaded config is a dataset config (missing core sections), merge with default.yaml
    required_sections = {"model", "policy", "detection", "fusion"}
    if not required_sections.issubset(loaded_config.keys()):
        default_config_path = PROJECT_ROOT / "configs" / "default.yaml"
        with open(default_config_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        
        if "data" in loaded_config and isinstance(loaded_config["data"], dict):
            base_config["data"].update(loaded_config["data"])
        elif "dataset" in loaded_config:
            base_config["data"].update(loaded_config)
            
        config = base_config
    else:
        config = loaded_config
    
    # Apply overrides
    if args.policies:
        config["policy"]["active_policies"] = args.policies
    if args.seeds:
        config["project"]["n_seeds"] = args.seeds
    
    # Determine output filename
    dataset_name = config.get("data", {}).get("dataset", "fraud")
    out_filename = args.output
    if not out_filename:
        out_filename = f"experiment_results_{dataset_name}.csv" if dataset_name not in ["fraud", "synthetic"] else "experiment_results.csv"

    # Set up MLflow if available
    if mlflow is not None:
        mlflow_cfg = config.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
        exp_name = mlflow_cfg.get("experiment_name", f"driftx_{dataset_name}")
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(exp_name)
        except Exception as e:
            logger.warning(f"MLflow setup warning: {e}")
    
    # Run experiments
    runner = ExperimentRunner(config)
    results = runner.run_all(output_filename=out_filename)
    
    print(f"\nExperiment complete! {len(results)} rows logged.")
    print(f"Results saved to {config.get('output', {}).get('results_dir', 'results')}/{out_filename}")
    
    # Print summary
    if not results.empty and "policy_name" in results.columns:
        summary = results.groupby("policy_name").agg({
            "accuracy": ["mean", "std"],
            "f1": ["mean", "std"],
            "cumulative_cost": "last",
            "cumulative_retrains": "last",
        })
        print(f"\n=== Experiment Summary ({dataset_name}) ===")
        print(summary)


if __name__ == "__main__":
    main()
