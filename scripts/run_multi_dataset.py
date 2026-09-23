"""Multi-Dataset Experiment Orchestrator for Phase 8 Generalization Proof."""
import sys
import logging
import argparse
import yaml
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner
from scripts.generate_tables import generate_cross_dataset_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RunMultiDataset")

DATASET_CONFIGS = {
    "Fraud": "configs/default.yaml",
    "Intrusion": "configs/datasets/intrusion.yaml",
    "Electricity": "configs/datasets/electricity.yaml",
}


def run_multi_dataset(
    datasets: List[str] = None,
    policies: List[str] = None,
    n_seeds: int = 1,
) -> Dict[str, Path]:
    """
    Run experiments across multiple datasets and generate a cross-dataset comparison table.

    Args:
        datasets: List of datasets to run ('Fraud', 'Intrusion', 'Electricity').
        policies: Policies to evaluate (defaults to ['p0_never', 'p2_drift_only', 'p5_dis_fused']).
        n_seeds: Number of random seeds per policy.

    Returns:
        Dict mapping dataset name to results CSV path.
    """
    if datasets is None:
        datasets = ["Fraud", "Intrusion", "Electricity"]
    if policies is None:
        policies = ["p0_never", "p2_drift_only", "p5_dis_fused"]

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_map = {}

    default_config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(default_config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    for ds_name in datasets:
        cfg_rel_path = DATASET_CONFIGS.get(ds_name)
        if not cfg_rel_path:
            logger.warning(f"Unknown dataset '{ds_name}'. Skipping.")
            continue

        cfg_path = PROJECT_ROOT / cfg_rel_path
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded_cfg = yaml.safe_load(f)

        import copy
        ds_config = copy.deepcopy(base_config)
        if "data" in loaded_cfg and isinstance(loaded_cfg["data"], dict):
            ds_config["data"].update(loaded_cfg["data"])
        elif "dataset" in loaded_cfg:
            ds_config["data"].update(loaded_cfg)

        ds_config["policy"]["active_policies"] = policies
        ds_config["project"]["n_seeds"] = n_seeds

        slug = ds_name.lower()
        out_filename = f"experiment_results_{slug}.csv"
        out_path = results_dir / out_filename

        logger.info(f"\n{'='*70}\nRunning Multi-Dataset Benchmark on: {ds_name} ({len(policies)} policies, {n_seeds} seeds)\n{'='*70}")
        runner = ExperimentRunner(ds_config)
        runner.run_all(output_filename=out_filename)
        results_map[ds_name] = out_path

    # Generate cross-dataset summary table
    logger.info("\nGenerating Cross-Dataset Generalization Table...")
    generate_cross_dataset_table(dataset_results_map=results_map, output_dir=str(results_dir))

    return results_map


def main():
    parser = argparse.ArgumentParser(description="DRIFT-X Multi-Dataset Runner")
    parser.add_argument("--datasets", nargs="+", default=["Fraud", "Intrusion", "Electricity"],
                        help="List of datasets to benchmark")
    parser.add_argument("--policies", nargs="+", default=["p0_never", "p2_drift_only", "p5_dis_fused"],
                        help="Policies to run across datasets")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds")
    args = parser.parse_args()

    run_multi_dataset(datasets=args.datasets, policies=args.policies, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
