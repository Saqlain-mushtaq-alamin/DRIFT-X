"""Visualization module for experiment results."""
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Style setup with fallback
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('default')

COLORS = {
    'p0_never': '#95a5a6',
    'p1_fixed': '#3498db',
    'p2_drift_only': '#e74c3c',
    'p3_shap_magnitude': '#f39c12',
    'p4_shap_rank': '#9b59b6',
    'p5_dis_fused': '#2ecc71',
}

POLICY_ORDER = [
    'p0_never', 'p1_fixed', 'p2_drift_only',
    'p3_shap_magnitude', 'p4_shap_rank', 'p5_dis_fused'
]


def load_results(results_dir: Union[str, Path] = "results") -> pd.DataFrame:
    """Load experiment results CSV."""
    path = Path(results_dir) / "experiment_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found at {path}")
    return pd.read_csv(path)


def plot_accuracy_over_time(
    df: pd.DataFrame, 
    output_dir: Union[str, Path] = "results/figures",
    metric: str = "accuracy"
):
    """
    PAPER FIGURE 1: Accuracy/F1 over time, one line per policy.
    Shows mean ± std across seeds.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for policy in POLICY_ORDER:
        policy_df = df[df["policy"] == policy]
        if policy_df.empty:
            continue
        
        # Group by window, compute mean ± std across seeds
        grouped = policy_df.groupby("window_id")[metric].agg(["mean", "std"])
        
        label = policy_df["policy_name"].iloc[0] if "policy_name" in policy_df.columns else policy
        color = COLORS.get(policy, '#333333')
        
        std_vals = grouped["std"].fillna(0.0)
        
        ax.plot(grouped.index, grouped["mean"], 
                label=label, color=color, linewidth=2.5, marker='o', markersize=4)
        ax.fill_between(
            grouped.index,
            grouped["mean"] - std_vals,
            grouped["mean"] + std_vals,
            alpha=0.15, color=color
        )
    
    ax.set_xlabel("Temporal Window Index", fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
    ax.set_title(f"Model Performance ({metric.capitalize()}) Over Temporal Windows", fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(out_path / f"{metric}_over_time.png", dpi=300)
    plt.savefig(out_path / f"{metric}_over_time.pdf")
    plt.close(fig)
    logger.info(f"Saved {metric}_over_time.png and .pdf")


def plot_cost_accuracy_tradeoff(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "results/figures"
):
    """
    PAPER FIGURE 2 (HEADLINE FIGURE): Accuracy vs Cost tradeoff.
    Each point is one policy. Shows the Pareto frontier.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    summary = []
    for policy in POLICY_ORDER:
        policy_df = df[df["policy"] == policy]
        if policy_df.empty:
            continue
        
        # Compute per-seed final metrics, then average
        per_seed = policy_df.groupby("seed").agg({
            "accuracy": "mean",
            "cumulative_cost": "last",
            "cumulative_retrains": "last",
        }).reset_index()
        
        mean_acc = per_seed["accuracy"].mean()
        std_acc = per_seed["accuracy"].std() if len(per_seed) > 1 else 0.0
        mean_cost = per_seed["cumulative_cost"].mean()
        std_cost = per_seed["cumulative_cost"].std() if len(per_seed) > 1 else 0.0
        mean_retrains = per_seed["cumulative_retrains"].mean()
        
        label = policy_df["policy_name"].iloc[0] if "policy_name" in policy_df.columns else policy
        color = COLORS.get(policy, '#333333')
        
        ax.errorbar(
            mean_cost, mean_acc,
            xerr=std_cost, yerr=std_acc,
            fmt='o', color=color, markersize=12,
            capsize=5, label=label, linewidth=2
        )
        
        # Annotate with retrain count
        ax.annotate(
            f'{mean_retrains:.1f} retrains',
            (mean_cost, mean_acc),
            textcoords="offset points",
            xytext=(10, -15), fontsize=9, fontweight='bold'
        )
        
        summary.append({
            "policy": policy,
            "policy_name": label,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "mean_retrains": mean_retrains,
        })
    
    ax.set_xlabel("Cumulative Retraining Cost (seconds)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Accuracy Across Windows", fontsize=12, fontweight='bold')
    ax.set_title("Accuracy vs. Cost Tradeoff (Pareto Frontier)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(out_path / "cost_accuracy_tradeoff.png", dpi=300)
    plt.savefig(out_path / "cost_accuracy_tradeoff.pdf")
    plt.close(fig)
    
    # Save summary table
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_path / "policy_summary.csv", index=False)
    logger.info("Saved cost_accuracy_tradeoff.png + policy_summary.csv")


def plot_retrain_timeline(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "results/figures"
):
    """
    PAPER FIGURE 3: When each policy retrains.
    Horizontal timeline with retrain markers.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    first_seed = df["seed"].iloc[0]
    seed_df = df[df["seed"] == first_seed]
    
    active_policies = [p for p in POLICY_ORDER if p in seed_df["policy"].unique()]
    
    for i, policy in enumerate(active_policies):
        policy_df = seed_df[seed_df["policy"] == policy]
        if policy_df.empty:
            continue
        
        retrain_windows = policy_df[policy_df["retrained"] == True]["window_id"]
        color = COLORS.get(policy, '#333333')
        label = policy_df["policy_name"].iloc[0] if "policy_name" in policy_df.columns else policy
        
        ax.scatter(
            retrain_windows, [i] * len(retrain_windows),
            color=color, s=120, marker='|', linewidth=3, label=label, zorder=3
        )
        ax.axhline(y=i, color=color, alpha=0.3, linewidth=1.5, linestyle=':')
    
    ax.set_xlabel("Temporal Window Index", fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(active_policies)))
    ax.set_yticklabels(
        [seed_df[seed_df["policy"] == p]["policy_name"].iloc[0] 
         if "policy_name" in seed_df.columns and not seed_df[seed_df["policy"] == p].empty else p
         for p in active_policies],
        fontsize=10, fontweight='bold'
    )
    ax.set_title("Retraining Event Timeline Across Policies", fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path / "retrain_timeline.png", dpi=300)
    plt.savefig(out_path / "retrain_timeline.pdf")
    plt.close(fig)
    logger.info("Saved retrain_timeline.png and .pdf")


def plot_dis_components(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "results/figures"
):
    """
    PAPER FIGURE 4: DIS components over time.
    Shows statistical drift, SHAP magnitude, and SHAP rank-change
    as stacked/overlaid signals.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    first_seed = df["seed"].iloc[0]
    p5_df = df[(df["policy"] == "p5_dis_fused") & (df["seed"] == first_seed)]
    
    if p5_df.empty:
        logger.warning("No P5 data found for DIS component plot")
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    windows = p5_df["window_id"]
    
    # Panel 1: Statistical drift
    axes[0].bar(windows, p5_df["stat_drift_score"], 
                color='#3498db', alpha=0.7, width=0.6)
    axes[0].set_ylabel("Statistical\nDrift Score", fontsize=10, fontweight='bold')
    axes[0].set_title("DIS Component Signal Breakdown & Adaptive Thresholding (P5: DIS-Fused)", fontsize=13, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: SHAP magnitude
    axes[1].bar(windows, p5_df["shap_magnitude_score"],
                color='#f39c12', alpha=0.7, width=0.6)
    axes[1].set_ylabel("SHAP\nMagnitude", fontsize=10, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Panel 3: SHAP rank-change
    axes[2].bar(windows, p5_df["shap_rank_change_score"],
                color='#9b59b6', alpha=0.7, width=0.6)
    axes[2].set_ylabel("SHAP Rank\nChange", fontsize=10, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    # Panel 4: Fused DIS + threshold
    axes[3].plot(windows, p5_df["dis"], 
                 color='#2ecc71', linewidth=2.5, label='DIS (Fused Score)')
    axes[3].plot(windows, p5_df["threshold"],
                 color='#e74c3c', linewidth=2, linestyle='--', 
                 label='Adaptive Threshold θ(t)')
    
    # Mark retrain points
    retrains = p5_df[p5_df["retrained"] == True]
    if not retrains.empty:
        axes[3].scatter(retrains["window_id"], retrains["dis"],
                        color='red', s=100, zorder=5, label='Retrain Event', marker='^')
    
    axes[3].set_ylabel("DIS / θ(t)", fontsize=10, fontweight='bold')
    axes[3].set_xlabel("Temporal Window Index", fontsize=12, fontweight='bold')
    axes[3].legend(loc="upper right", fontsize=10, frameon=True)
    axes[3].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path / "dis_components.png", dpi=300)
    plt.savefig(out_path / "dis_components.pdf")
    plt.close(fig)
    logger.info("Saved dis_components.png and .pdf")


def generate_all_figures(results_dir: Union[str, Path] = "results"):
    """Generate all paper figures from experiment results."""
    df = load_results(results_dir)
    out_dir = Path(results_dir) / "figures"
    
    plot_accuracy_over_time(df, out_dir, "accuracy")
    plot_accuracy_over_time(df, out_dir, "f1")
    plot_cost_accuracy_tradeoff(df, out_dir)
    plot_retrain_timeline(df, out_dir)
    plot_dis_components(df, out_dir)
    
    logger.info(f"All paper figures successfully saved to {out_dir}/")
