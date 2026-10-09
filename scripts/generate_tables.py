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
    'p3_shap_magnitude', 'p4_shap_rank', 'p5_dis_fused',
    'p6_performance_drop', 'p7_weighted_ks', 'p8_random_budget', 'p9_shap_loss'
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
        
        # Extract fusion hyperparameters from CSV if present (logged since Bug-6 fix).
        # Use dropna() to skip window-0 rows which don't carry these columns.
        config_str = "N/A"
        if all(c in p_df.columns for c in ("alpha", "beta", "gamma", "adaptive_lambda", "lookback_k")):
            param_rows = p_df.dropna(subset=["alpha", "beta", "gamma", "adaptive_lambda", "lookback_k"])
            if not param_rows.empty:
                p_row = param_rows.iloc[0]
                config_str = (
                    f"α={p_row['alpha']:.1f}, β={p_row['beta']:.1f}, "
                    f"γ={p_row['gamma']:.1f}, λ={p_row['adaptive_lambda']:.1f}, k={int(p_row['lookback_k'])}"
                )

        summary.append({
            "Policy": policy_name,
            "Accuracy": f"{mean_acc:.4f} ± {std_acc:.4f}",
            "F1 Score": f"{mean_f1:.4f} ± {std_f1:.4f}",
            "Cost (s)": f"{mean_cost:.2f}",
            "Retrains": f"{mean_retrains:.1f}",
            "Config (DIS-Fused only)": config_str if policy == "p5_dis_fused" else "—",
        })
    
    table_df = pd.DataFrame(summary)
    
    out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LaTeX table
    latex_str = table_df.to_latex(index=False, caption="DRIFT-X Policy Performance Comparison", label="tab:policy_results")
    latex_path = out_dir / "paper_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    # Save Markdown table with footnote
    footnote = (
        "\n> **Config footnote (DIS-Fused):** "
        "α=weight for statistical drift, β=weight for SHAP magnitude, "
        "γ=weight for SHAP rank-change, λ=ATC sensitivity, k=ATC lookback windows.\n"
    )
    md_str = table_df.to_markdown(index=False)
    md_path = out_dir / "paper_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_str + footnote)

    logger.info(f"✓ Saved LaTeX table to {latex_path}")
    logger.info(f"✓ Saved Markdown table to {md_path}")

    # Use safe encoding for Windows console which may not support UTF-8/Greek chars
    def _safe_print(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", errors="backslashreplace").decode("ascii"))

    _safe_print("\n=== DRIFT-X Publication Results Table ===")
    _safe_print(md_str)
    _safe_print(footnote)
    
    return table_df


def generate_cross_dataset_table(
    dataset_results_map: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate Cross-Dataset Results Table comparing P0, P2, and P5 performance across domains.

    Args:
        dataset_results_map: Dict mapping dataset name (e.g. 'Fraud', 'Intrusion', 'Electricity')
            to either a file path (str/Path) or a loaded pd.DataFrame.
        output_dir: Directory to save the output tables. Defaults to 'results/'.

    Returns:
        pd.DataFrame containing the cross-dataset comparison table.
    """
    results_base = Path(output_dir) if output_dir else PROJECT_ROOT / "results"
    results_base.mkdir(parents=True, exist_ok=True)

    if dataset_results_map is None:
        dataset_results_map = {}
        # Candidate paths
        candidates = {
            "Fraud": [results_base / "experiment_results_fraud.csv", results_base / "experiment_results.csv"],
            "Intrusion": [results_base / "experiment_results_intrusion.csv"],
            "Electricity": [results_base / "experiment_results_electricity.csv"],
        }
        for name, paths in candidates.items():
            for p in paths:
                if p.exists():
                    dataset_results_map[name] = p
                    break

    rows = []
    for ds_name, source in dataset_results_map.items():
        if isinstance(source, pd.DataFrame):
            df = source
        else:
            p = Path(source)
            if not p.exists():
                logger.warning(f"Results file for '{ds_name}' not found at {p}. Skipping.")
                continue
            df = pd.read_csv(p)

        def get_policy_stats(policy_id: str):
            p_df = df[df["policy"] == policy_id]
            if p_df.empty:
                return {"acc_str": "N/A", "retrains": "N/A", "cost": "N/A"}
            per_seed = p_df.groupby("seed").agg({
                "accuracy": "mean",
                "cumulative_cost": "last",
                "cumulative_retrains": "last",
            })
            m_acc = per_seed["accuracy"].mean()
            s_acc = per_seed["accuracy"].std() if len(per_seed) > 1 else 0.0
            m_cost = per_seed["cumulative_cost"].mean()
            m_retrains = per_seed["cumulative_retrains"].mean()
            return {
                "acc_str": f"{m_acc:.4f} ± {s_acc:.4f}" if len(per_seed) > 1 else f"{m_acc:.4f}",
                "retrains": f"{m_retrains:.1f}",
                "cost": f"{m_cost:.2f}s",
            }

        p0_stats = get_policy_stats("p0_never")
        p2_stats = get_policy_stats("p2_drift_only")
        p5_stats = get_policy_stats("p5_dis_fused")

        rows.append({
            "Dataset": ds_name,
            "P0 Acc": p0_stats["acc_str"],
            "P2 Acc": p2_stats["acc_str"],
            "P5 Acc": p5_stats["acc_str"],
            "P5 Retrains": p5_stats["retrains"],
            "P5 Cost": p5_stats["cost"],
        })

    cross_df = pd.DataFrame(rows)
    if cross_df.empty:
        logger.warning("No dataset results available to compile cross-dataset table.")
        return cross_df

    # Save LaTeX table
    latex_str = cross_df.to_latex(
        index=False,
        caption="Cross-Dataset Multi-Domain Generalization Benchmark (DRIFT-X)",
        label="tab:cross_dataset_results",
    )
    latex_path = results_base / "cross_dataset_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    # Save Markdown table
    md_str = cross_df.to_markdown(index=False)
    md_path = results_base / "cross_dataset_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_str + "\n")

    logger.info(f"✓ Saved Cross-Dataset LaTeX table to {latex_path}")
    logger.info(f"✓ Saved Cross-Dataset Markdown table to {md_path}")

    def _safe_print(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", errors="backslashreplace").decode("ascii"))

    _safe_print("\n=== DRIFT-X Cross-Dataset Generalization Results ===")
    _safe_print(md_str)

    return cross_df


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_paper_tables(path)
    generate_cross_dataset_table()

