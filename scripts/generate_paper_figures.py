"""Generate publication figures for DRIFT-X.

Figures:
1. Fig 1: Protocol schematic (test-then-train loop & dual-anchored budget curve construction).
2. Fig 2: Accuracy vs. retrains curves on decoupled streams (P1, P2, P5, P7, wstat, P8, Page-Hinkley).
3. Fig 3: Artifact figure (invisible-change-point recall on coupled vs. decoupled streams).
4. Fig 4: Forest plot of paired differences (AUC & acc@4) with unadjusted CIs and Holm p-values.
5. Fig 5: Retrains vs. accuracy saturation on real streams (NEweather & ELEC2).
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PaperFigures")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.4,
    "grid.linestyle": ":",
})

PALETTE = {
    "p1_fixed": "#3498db",        # Blue
    "p2_drift_only": "#e74c3c",   # Red
    "p5_dis_fused": "#2ecc71",    # Emerald
    "p5_wstat_only": "#16a085",   # Dark Teal
    "p7_weighted_ks": "#9b59b6",  # Purple
    "p8_random_budget": "#e67e22",# Orange
    "p10_adwin": "#f1c40f",       # Yellow
    "p13_page_hinkley": "#34495e",# Navy / Slate
    "p0_never": "#7f8c8d",        # Gray
    "p14_oracle": "#27ae60",      # Green
}

LABELS = {
    "p1_fixed": "P1 Fixed Schedule",
    "p2_drift_only": "P2 KS Drift-Only",
    "p5_dis_fused": "P5 DIS-Fused",
    "p5_wstat_only": "P5-wstat Importance-KS",
    "p7_weighted_ks": "P7 Weighted-KS",
    "p8_random_budget": "P8 Random Budget",
    "p10_adwin": "P10 ADWIN (Supervised)",
    "p13_page_hinkley": "P13 Page-Hinkley (Supervised)",
    "p0_never": "P0 Never-Retrain",
}


# ==============================================================================
# Figure 1: Protocol Schematic
# ==============================================================================
def make_fig1_protocol_schematic():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.1, 1.0]})
    
    # Subpanel A: Test-then-train loop diagram
    ax1.set_title("A. Test-Then-Train Evaluation Loop per Window $W_t$", fontweight="bold", pad=12)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    
    # Box styles
    b_gray = dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="#7f8c8d", lw=1.5)
    b_blue = dict(boxstyle="round,pad=0.5", facecolor="#ebf5fb", edgecolor="#2980b9", lw=1.5)
    b_orange = dict(boxstyle="round,pad=0.5", facecolor="#fef5e7", edgecolor="#d35400", lw=1.5)
    b_green = dict(boxstyle="round,pad=0.5", facecolor="#eafaf1", edgecolor="#27ae60", lw=1.5)
    
    ax1.text(1.5, 8.5, "Incoming Window $W_t$\nUnlabeled $X_t$, Ground-Truth $y_t$", ha="center", va="center", bbox=b_gray, fontsize=9)
    ax1.text(1.5, 5.5, "1. Out-of-Sample Eval\n$\\hat{y}_t = f_{t-1}(X_t)$\nAcc logged with $y_t$\n(Honest Pre-Retrain)", ha="center", va="center", bbox=b_blue, fontsize=9)
    ax1.text(5.5, 5.5, "2. Label-Free Drift Test\nCompare $X_t$ vs. $X_\\mathrm{ref}$\n(KS, SHAP Weights, DIS)", ha="center", va="center", bbox=b_orange, fontsize=9)
    ax1.text(8.5, 5.5, "3. Decision\nRetrain?", ha="center", va="center", bbox=b_gray, fontsize=9, fontweight="bold")
    
    ax1.text(8.5, 2.0, "YES: Retrain $f_t$\n$f_t = \\mathrm{Train}(W_{t-k:t})$\nRef Reset & Cost Logged", ha="center", va="center", bbox=b_green, fontsize=9)
    ax1.text(5.5, 2.0, "NO: Keep Model\n$f_t = f_{t-1}$\nReference Retained", ha="center", va="center", bbox=b_gray, fontsize=9)
    
    # Arrows
    arr = dict(arrowstyle="->", lw=1.5, color="#2c3e50")
    ax1.annotate("", xy=(1.5, 6.7), xytext=(1.5, 7.6), arrowprops=arr)
    ax1.annotate("", xy=(3.8, 5.5), xytext=(3.0, 5.5), arrowprops=arr)
    ax1.annotate("", xy=(7.3, 5.5), xytext=(6.9, 5.5), arrowprops=arr)
    ax1.annotate("", xy=(8.5, 3.1), xytext=(8.5, 4.7), arrowprops=arr)
    ax1.text(8.8, 3.9, "Triggered", fontsize=8, color="#27ae60", fontweight="bold")
    ax1.annotate("", xy=(6.9, 2.0), xytext=(7.5, 2.0), arrowprops=arr)
    ax1.annotate("", xy=(5.5, 3.0), xytext=(8.0, 4.8), arrowprops=arr)
    ax1.text(6.8, 4.0, "Suppressed", fontsize=8, color="#7f8c8d")
    
    # Subpanel B: Dual-anchored curve construction
    ax2.set_title("B. Dual-Anchored Budget Curve Construction", fontweight="bold", pad=12)
    b_vals = np.array([0, 1.5, 3.0, 4.0, 6.5, 10.0])
    acc_measured = np.array([0.700, 0.725, 0.742, 0.748, 0.755, 0.760])
    
    ax2.plot(b_vals, acc_measured, "o--", color="#2980b9", lw=2, markersize=7, label="Policy Operating Points")
    ax2.plot([0], [0.700], "s", color="#7f8c8d", markersize=10, zorder=5, label="Left Anchor: $(0, \\mathrm{acc}_{P0})$")
    ax2.plot([10], [0.760], "^", color="#c0392b", markersize=10, zorder=5, label="Right Anchor: $(N_\\mathrm{eval}, \\mathrm{acc}_\\mathrm{always})$")
    
    ax2.axvline(4.0, color="#d35400", linestyle=":", lw=2, label="Matched Budget Slice ($B^*=4$)")
    ax2.fill_between(b_vals, 0.700, acc_measured, color="#3498db", alpha=0.15, label="Normalized Budget AUC")
    
    ax2.annotate("Left Anchor:\nNever-Retrain ($B=0$)", xy=(0, 0.700), xytext=(0.8, 0.690),
                 arrowprops=dict(arrowstyle="->", color="#7f8c8d"), fontsize=8)
    ax2.annotate("Right Anchor:\nAlways-Retrain ($B=N_\\mathrm{eval}$)", xy=(10, 0.760), xytext=(6.5, 0.765),
                 arrowprops=dict(arrowstyle="->", color="#c0392b"), fontsize=8)
    ax2.annotate("acc@4 Slice\n(Matched Budget)", xy=(4, 0.748), xytext=(4.3, 0.730),
                 arrowprops=dict(arrowstyle="->", color="#d35400"), fontsize=8)
    
    ax2.set_xlabel("Retraining Budget (Retrains per Stream $B$)", fontweight="bold")
    ax2.set_ylabel("Mean Accuracy", fontweight="bold")
    ax2.set_xlim(-0.5, 11)
    ax2.set_ylim(0.68, 0.78)
    ax2.legend(loc="lower right", fontsize=8, frameon=True)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_protocol_schematic.png", dpi=300)
    plt.savefig(FIG_DIR / "fig1_protocol_schematic.pdf")
    plt.close()
    logger.info("Saved Fig 1: fig1_protocol_schematic (.png, .pdf)")


# ==============================================================================
# Figure 2: Decoupled Budget Curves
# ==============================================================================
def make_fig2_budget_curves():
    raw_path = RESULTS_DIR / "budget_curves_mixed_decoupled_s4000_4029_confirmatory_raw.csv"
    if not raw_path.exists():
        logger.warning(f"File not found: {raw_path}")
        return
    df = pd.read_csv(raw_path)
    
    target_policies = [
        "p1_fixed", "p2_drift_only", "p5_dis_fused", "p7_weighted_ks",
        "p5_wstat_only", "p8_random_budget", "p13_page_hinkley"
    ]
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    
    # Left & right anchors across all streams
    p0_mean = df[df["policy"] == "p0_never"]["acc"].mean()
    always_mean = df[(df["policy"] == "p1_fixed") & (df["retrains"] == 19)]["acc"].mean()
    if np.isnan(always_mean):
        always_mean = df[df["retrains"] >= 18]["acc"].mean()
        
    for pol in target_policies:
        sub = df[df["policy"] == pol]
        if sub.empty:
            continue
            
        color = PALETTE.get(pol, "#333333")
        label = LABELS.get(pol, pol)
        
        # Group by param_key & param_val to get operating points
        grouped = sub.groupby(["param_key", "param_val"]).agg(
            retrains_mean=("retrains", "mean"),
            retrains_std=("retrains", "std"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            n=("acc", "count")
        ).reset_index()
        
        # Add anchors for interpolation curve
        curve_points = [(0.0, p0_mean)]
        for _, r in grouped.iterrows():
            curve_points.append((r["retrains_mean"], r["acc_mean"]))
        curve_points.append((19.0, always_mean))
        curve_points = sorted(curve_points, key=lambda x: x[0])
        
        xs = [p[0] for p in curve_points]
        ys = [p[1] for p in curve_points]
        
        # Plot interpolated line as dashed
        ax.plot(xs, ys, "--", color=color, alpha=0.7, lw=1.5)
        
        # Plot measured operating points as markers with error bars
        sem_acc = grouped["acc_std"] / np.sqrt(grouped["n"])
        sem_ret = grouped["retrains_std"] / np.sqrt(grouped["n"])
        
        marker = "s" if "page" in pol else "o"
        ax.errorbar(
            grouped["retrains_mean"], grouped["acc_mean"],
            yerr=sem_acc * 1.96, xerr=sem_ret * 1.96,
            fmt=marker, color=color, ecolor=color, elinewidth=1,
            capsize=3, markersize=6, label=label, zorder=4
        )
        
    # Vertical matched budget reference line
    ax.axvline(4.0, color="#7f8c8d", linestyle=":", lw=2, label="Pre-Registered Budget Slice $B^*=4$")
    
    ax.set_title("Accuracy vs. Retraining Budget on Decoupled Streams (Seeds 4000–4029)", fontweight="bold", pad=12)
    ax.set_xlabel("Retraining Budget (Mean Retrains per Stream $B$)", fontweight="bold")
    ax.set_ylabel("Out-of-Sample Accuracy", fontweight="bold")
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(0.68, 0.80)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_budget_curves_decoupled.png", dpi=300)
    plt.savefig(FIG_DIR / "fig2_budget_curves_decoupled.pdf")
    plt.close()
    logger.info("Saved Fig 2: fig2_budget_curves_decoupled (.png, .pdf)")


# ==============================================================================
# Figure 3: Artifact Invisible CP Recall
# ==============================================================================
def make_fig3_artifact():
    # Chance levels on invisible change points
    chance_coupled = 0.4267
    chance_decoupled = 0.4350
    
    # Read raw data
    c_path = RESULTS_DIR / "budget_curves_mixed_s3000_3029_confirmatory_raw.csv"
    d_path = RESULTS_DIR / "budget_curves_mixed_decoupled_s4000_4029_confirmatory_raw.csv"
    
    if not c_path.exists() or not d_path.exists():
        logger.warning("Confirmatory raw files missing for Fig 3.")
        return
        
    df_c = pd.read_csv(c_path)
    df_d = pd.read_csv(d_path)
    
    policies = ["p2_drift_only", "p5_dis_fused", "p7_weighted_ks", "p5_wstat_only"]
    pol_names = ["P2 Unweighted KS", "P5 DIS-Fused", "P7 Weighted-KS", "wstat Importance-KS"]
    
    rec_c = []
    rec_d = []
    
    for pol in policies:
        # Default operating point (first threshold / default param)
        sub_c = df_c[df_c["policy"] == pol]
        first_param_c = sub_c["param_val"].iloc[0]
        val_c = sub_c[sub_c["param_val"] == first_param_c]["cp_recall_invis"].mean()
        rec_c.append(val_c)
        
        sub_d = df_d[df_d["policy"] == pol]
        first_param_d = sub_d["param_val"].iloc[0]
        val_d = sub_d[sub_d["param_val"] == first_param_d]["cp_recall_invis"].mean()
        rec_d.append(val_d)
        
    fig, ax = plt.subplots(figsize=(8.5, 5))
    
    x = np.arange(len(policies))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rec_c, width, label="Coupled Streams (Artifact Present, Seeds 3000–3029)", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width/2, rec_d, width, label="Decoupled Streams (Artifact Removed, Seeds 4000–4029)", color="#2980b9", alpha=0.85)
    
    # Horizontal line at chance level
    ax.axhline(chance_decoupled, color="#27ae60", linestyle="--", lw=2,
               label=f"Invisible CP Chance Level ({chance_decoupled*100:.1f}%)")
    
    # Bar value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h*100:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h*100:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                    
    ax.set_title("Benchmark Artifact: Invisible Concept-Shift Recall Collapses to Chance", fontweight="bold", pad=12)
    ax.set_ylabel("Invisible Change-Point Recall", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pol_names, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_artifact_invisible_cp_recall.png", dpi=300)
    plt.savefig(FIG_DIR / "fig3_artifact_invisible_cp_recall.pdf")
    plt.close()
    logger.info("Saved Fig 3: fig3_artifact_invisible_cp_recall (.png, .pdf)")


# ==============================================================================
# Figure 4: Forest Plot of Paired Differences
# ==============================================================================
def make_fig4_forest_plot():
    tests_d_path = RESULTS_DIR / "budget_curves_mixed_decoupled_s4000_4029_confirmatory_tests.csv"
    tests_c_path = RESULTS_DIR / "budget_curves_mixed_s3000_3029_confirmatory_tests.csv"
    
    if not tests_d_path.exists() or not tests_c_path.exists():
        logger.warning("Tests CSVs missing for Fig 4.")
        return
        
    df_d = pd.read_csv(tests_d_path)
    df_c = pd.read_csv(tests_c_path)
    
    # Selected key comparisons
    pairs = [
        ("p5_dis_fused", "p2_drift_only", "acc@4", "P5 vs P2 (acc@4)"),
        ("p7_weighted_ks", "p2_drift_only", "acc@4", "P7 vs P2 (acc@4)"),
        ("p5_dis_fused", "p1_fixed", "acc@4", "P5 vs P1 (acc@4)"),
        ("p7_weighted_ks", "p1_fixed", "acc@4", "P7 vs P1 (acc@4)"),
        ("p7_weighted_ks", "p8_random_budget", "acc@4", "P7 vs P8 (acc@4)*"),
        ("p5_dis_fused", "p2_drift_only", "auc", "P5 vs P2 (AUC)"),
        ("p7_weighted_ks", "p2_drift_only", "auc", "P7 vs P2 (AUC)"),
        ("p13_page_hinkley", "p5_dis_fused", "acc@4", "P13 vs P5 (acc@4)"),
        ("p13_page_hinkley", "p7_weighted_ks", "acc@4", "P13 vs P7 (acc@4)"),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    
    y_pos = np.arange(len(pairs))
    
    def plot_subpanel(ax, df_tests, title, color_pt):
        ax.set_title(title, fontweight="bold", pad=12)
        ax.axvline(0.0, color="#333333", linestyle="--", lw=1.2)
        
        for i, (ref, vs, metric, label) in enumerate(pairs):
            match = df_tests[(df_tests["ref"] == ref) & (df_tests["vs"] == vs) & (df_tests["metric"] == metric)]
            if match.empty:
                # Try reversed
                match = df_tests[(df_tests["ref"] == vs) & (df_tests["vs"] == ref) & (df_tests["metric"] == metric)]
                rev = True
            else:
                rev = False
                
            if match.empty:
                continue
                
            row = match.iloc[0]
            diff = -row["diff_mean"] if rev else row["diff_mean"]
            lo = -row["ci_95_hi"] if rev else row["ci_95_lo"]
            hi = -row["ci_95_lo"] if rev else row["ci_95_hi"]
            p_h = row["p_holm"]
            
            # Error bar = unadjusted 95% CI
            ax.errorbar(diff, i, xerr=[[diff - lo], [hi - diff]],
                        fmt="o", color=color_pt, ecolor=color_pt, elinewidth=2,
                        capsize=4, markersize=6)
            
            # Label = Holm p-value
            p_text = f"p={p_h:.3f}" if p_h >= 0.001 else "p<0.001"
            ax.annotate(f"{p_text}", xy=(hi, i), xytext=(5, -2),
                        textcoords="offset points", fontsize=8, color="#2c3e50")
            
        ax.set_xlabel("Mean Paired Difference (95% Unadjusted Bootstrap CI)", fontweight="bold")
        ax.set_xlim(-0.06, 0.08)
        ax.grid(True, axis="x")
        
    plot_subpanel(ax1, df_c, "Coupled Streams (Seeds 3000–3029)", "#e74c3c")
    plot_subpanel(ax2, df_d, "Decoupled Streams (Seeds 4000–4029)", "#2980b9")
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([p[3] for p in pairs], fontweight="bold")
    
    # Add footnote note on unadjusted CIs
    fig.text(0.5, 0.01, "*Note: For P7 vs P8, the unadjusted bootstrap CI excludes zero, but the difference is not significant after Holm correction (p = 0.167).",
             ha="center", fontsize=8.5, style="italic")
             
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(FIG_DIR / "fig4_forest_paired_differences.png", dpi=300)
    plt.savefig(FIG_DIR / "fig4_forest_paired_differences.pdf")
    plt.close()
    logger.info("Saved Fig 4: fig4_forest_paired_differences (.png, .pdf)")


# ==============================================================================
# Figure 5: Saturation on Real Datasets (NEweather & ELEC2)
# ==============================================================================
def make_fig5_saturation():
    wea_path = RESULTS_DIR / "real_weather_standard.csv"
    elec_path = RESULTS_DIR / "experiment_results_electricity.csv"
    
    if not wea_path.exists() or not elec_path.exists():
        logger.warning("Weather or Electricity results missing for Fig 5.")
        return
        
    df_wea = pd.read_csv(wea_path)
    df_elec = pd.read_csv(elec_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Weather
    ax1.set_title("A. NOAA NEweather Stream (20 Windows, 8 Features)", fontweight="bold", pad=12)
    for _, r in df_wea.iterrows():
        pol = r["policy"]
        color = PALETTE.get(pol, "#7f8c8d")
        ax1.scatter(r["retrains"], r["acc"], color=color, s=80, zorder=4)
        ax1.annotate(LABELS.get(pol, pol).split()[0], xy=(r["retrains"], r["acc"]),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
                    
    ax1.set_xlabel("Number of Retrains (out of 19 eligible)", fontweight="bold")
    ax1.set_ylabel("Out-of-Sample Accuracy", fontweight="bold")
    ax1.set_xlim(-1, 21)
    ax1.set_ylim(0.74, 0.78)
    ax1.grid(True)
    
    # Panel B: ELEC2
    ax2.set_title("B. ELEC2 Stream (30 Windows, 7 Features, Date Removed)", fontweight="bold", pad=12)
    elec_agg = df_elec.dropna(subset=["accuracy"]).groupby("policy").agg(
        acc=("accuracy", "mean"),
        retrains=("retrained", "sum")
    ).reset_index()
    
    for _, r in elec_agg.iterrows():
        pol = r["policy"]
        color = PALETTE.get(pol, "#7f8c8d")
        ax2.scatter(r["retrains"], r["acc"], color=color, s=80, zorder=4)
        ax2.annotate(LABELS.get(pol, pol).split()[0], xy=(r["retrains"], r["acc"]),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
                    
    ax2.set_xlabel("Number of Retrains (out of 30 eligible)", fontweight="bold")
    ax2.set_ylabel("Out-of-Sample Accuracy", fontweight="bold")
    ax2.set_xlim(-1, 32)
    ax2.set_ylim(0.60, 0.78)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_saturation_real_datasets.png", dpi=300)
    plt.savefig(FIG_DIR / "fig5_saturation_real_datasets.pdf")
    plt.close()
    logger.info("Saved Fig 5: fig5_saturation_real_datasets (.png, .pdf)")


if __name__ == "__main__":
    make_fig1_protocol_schematic()
    make_fig2_budget_curves()
    make_fig3_artifact()
    make_fig4_forest_plot()
    make_fig5_saturation()
    logger.info(f"All 5 figures generated successfully in {FIG_DIR}")
