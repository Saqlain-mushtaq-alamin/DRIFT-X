"""Generate LaTeX and Markdown tables for paper publication."""
import sys
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GenerateTables")

POLICY_ORDER = [
    'p0_never', 'p1_fixed', 'p2_drift_only',
    'p3_shap_magnitude', 'p4_shap_rank', 'p5_dis_fused'
]


def generate_paper_tables(results_path: Optional[str] = None):
    if results_path is None:
        csv_path = PROJECT_ROOT / "results" / "experiment_results.csv"
    else:
        csv_path = Path(results_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Experiment results CSV missing at {csv_path}")
    
    df = pd.read_csv(csv_path)
    summary = []
    
    for policy in POLICY_ORDER:
        p_df = df[df["policy"] == policy]
        if p_df.empty:
            continue
        
        per_seed = p_df.groupby("seed").agg({
            "accuracy": "mean",
            "f1": "mean",
            "cumulative_cost": "last",
            "cumulative_retrains": "last",
        })
        
        policy_name = p_df["policy_name"].iloc[0] if "policy_name" in p_df.columns else policy
        
        mean_acc = per_seed["accuracy"].mean()
        std_acc = per_seed["accuracy"].std() if len(per_seed) > 1 else 0.0
        mean_f1 = per_seed["f1"].mean()
        std_f1 = per_seed["f1"].std() if len(per_seed) > 1 else 0.0
        mean_cost = per_seed["cumulative_cost"].mean()
        mean_retrains = per_seed["cumulative_retrains"].mean()
        
        summary.append({
            "Policy": policy_name,
            "Accuracy": f"{mean_acc:.4f} ± {std_acc:.4f}",
            "F1 Score": f"{mean_f1:.4f} ± {std_f1:.4f}",
            "Cost (s)": f"{mean_cost:.2f}",
            "Retrains": f"{mean_retrains:.1f}",
        })
    
    table_df = pd.DataFrame(summary)
    
    out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LaTeX table
    latex_str = table_df.to_latex(index=False, caption="DRIFT-X Policy Performance Comparison", label="tab:policy_results")
    latex_path = out_dir / "paper_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    # Save Markdown table
    md_str = table_df.to_markdown(index=False)
    md_path = out_dir / "paper_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_str)
    
    logger.info(f"✓ Saved LaTeX table to {latex_path}")
    logger.info(f"✓ Saved Markdown table to {md_path}")
    
    print("\n=== DRIFT-X Publication Results Table ===")
    print(md_str)
    
    return table_df


if __name__ == "__main__":
    generate_paper_tables()
