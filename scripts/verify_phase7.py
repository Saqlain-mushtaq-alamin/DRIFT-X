"""Phase 7 Experiments, Ablation & Visualization Verification Script."""
import sys
import logging
import yaml
import copy
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner
from driftx.viz.plotter import generate_all_figures
from scripts.run_ablation import ABLATION_CONFIGS, run_ablation
from scripts.run_sensitivity import run_sensitivity
from scripts.generate_tables import generate_paper_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase7_Verification")


def run_verification():
    logger.info("=== Phase 7 Experiments & Visualization Verification Script ===")

    # 1. Load configuration and run experiment runner (2 seeds)
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["project"]["n_seeds"] = 2
    logger.info("Step 1: Running ExperimentRunner (2 seeds)...")
    runner = ExperimentRunner(config)
    results = runner.run_all()
    assert not results.empty, "Experiment results dataframe is empty!"
    logger.info("✓ Step 1 Complete: Experiment runner finished.")

    # 2. Generate all paper figures
    logger.info("\nStep 2: Generating all paper figures (PNG & PDF)...")
    results_dir = PROJECT_ROOT / "results"
    generate_all_figures(results_dir=results_dir)
    
    figures_dir = results_dir / "figures"
    expected_figures = [
        "accuracy_over_time.png", "accuracy_over_time.pdf",
        "f1_over_time.png", "f1_over_time.pdf",
        "cost_accuracy_tradeoff.png", "cost_accuracy_tradeoff.pdf",
        "retrain_timeline.png", "retrain_timeline.pdf",
        "dis_components.png", "dis_components.pdf",
        "policy_summary.csv"
    ]
    for fig_file in expected_figures:
        fig_path = figures_dir / fig_file
        assert fig_path.exists(), f"Expected figure/summary artifact missing: {fig_path}"
        logger.info(f"  ✓ Found figure artifact: {fig_file}")
    logger.info("✓ Step 2 Complete: All paper figures generated successfully.")

    # 3. Fast Ablation Run Verification
    logger.info("\nStep 3: Verifying Ablation Study execution...")
    ablation_csv = results_dir / "ablation_results.csv"
    ablation_df = run_ablation(config_path=str(config_path))
    assert ablation_csv.exists(), f"Ablation results CSV missing at {ablation_csv}"
    assert not ablation_df.empty, "Ablation results dataframe is empty!"
    logger.info("✓ Step 3 Complete: Ablation study executed and verified.")

    # 4. Generate LaTeX and Markdown Paper Tables
    logger.info("\nStep 4: Generating Paper LaTeX and Markdown Tables...")
    table_df = generate_paper_tables(results_path=str(results_dir / "experiment_results.csv"))
    assert not table_df.empty, "Paper table dataframe is empty!"
    assert (results_dir / "paper_table.tex").exists(), "LaTeX table paper_table.tex missing!"
    assert (results_dir / "paper_table.md").exists(), "Markdown table paper_table.md missing!"
    logger.info("✓ Step 4 Complete: LaTeX and Markdown publication tables generated.")

    logger.info("\n✓ All Phase 7 Verification Checks PASSED Successfully!")


if __name__ == "__main__":
    run_verification()
