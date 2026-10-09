import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ResultsSummary")

content = []
content.append("# DRIFT-X Comprehensive Experiment Results Summary\n")

# Verify config hashes across output files
hashes = {}
csv_files_to_check = [
    "results/experiment_results.csv",
    "results/changepoint_results.csv",
    "results/changepoint_concept_only_results.csv",
    "results/null_stream_results.csv",
    "results/null_stream_correlated_results.csv",
    "results/null_stream_elec2_null_results.csv",
    "results/ablation_results.csv",
]

for p_str in csv_files_to_check:
    p = Path(p_str)
    if p.exists():
        try:
            df_check = pd.read_csv(p, nrows=50)
            if "config_hash" in df_check.columns:
                h_vals = df_check["config_hash"].dropna().unique()
                if len(h_vals) > 0:
                    hashes[p.name] = h_vals[0]
        except Exception:
            pass

content.append("## Provenance & Reproducibility Hashes\n")
if hashes:
    hash_df = pd.DataFrame([{"File": k, "Config Hash": v} for k, v in hashes.items()])
    content.append(hash_df.to_markdown(index=False))
    unique_hashes = set(hashes.values())
    if len(unique_hashes) > 1:
        content.append(f"\n> **Note:** Multiple config hashes detected: {unique_hashes}\n")
    else:
        content.append(f"\n> **Verified:** All generated results share uniform config hash: `{list(unique_hashes)[0]}`\n")
else:
    content.append("No config hashes recorded in result files.\n")
content.append("\n")

# 1. Main Benchmark (experiment_results.csv)
p1 = Path("results/experiment_results.csv")
if p1.exists():
    df = pd.read_csv(p1)
    eval_df = df.dropna(subset=["accuracy"])
    per_seed = eval_df.groupby(["policy", "policy_name", "seed"]).agg(
        acc=("accuracy", "mean"),
        f1=("f1", "mean"),
        retrains=("retrained", "sum"),
        cost=("cost_seconds", "sum")
    ).reset_index()
    summary = per_seed.groupby(["policy", "policy_name"]).agg(
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        retrains_mean=("retrains", "mean"),
        retrains_std=("retrains", "std"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std")
    ).round(4).reset_index()

    content.append("## 1. Main Benchmark — All 10 Policies (results/experiment_results.csv)\n")
    content.append(summary.to_markdown(index=False))
    content.append("\n")

# 2. Dataset: Fraud (experiment_results_fraud.csv)
p2 = Path("results/experiment_results_fraud.csv")
if p2.exists():
    df = pd.read_csv(p2)
    eval_df = df.dropna(subset=["accuracy"])
    per_seed = eval_df.groupby(["policy", "policy_name", "seed"]).agg(
        acc=("accuracy", "mean"),
        f1=("f1", "mean"),
        retrains=("retrained", "sum"),
        cost=("cost_seconds", "sum")
    ).reset_index()
    summary = per_seed.groupby(["policy", "policy_name"]).agg(
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        retrains_mean=("retrains", "mean"),
        retrains_std=("retrains", "std"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std")
    ).round(4).reset_index()

    content.append("## 2. Fraud Stream (results/experiment_results_fraud.csv)\n")
    content.append(summary.to_markdown(index=False))
    content.append("\n")

# 3. Dataset: Intrusion (experiment_results_intrusion.csv)
p3 = Path("results/experiment_results_intrusion.csv")
if p3.exists():
    df = pd.read_csv(p3)
    eval_df = df.dropna(subset=["accuracy"])
    per_seed = eval_df.groupby(["policy", "policy_name", "seed"]).agg(
        acc=("accuracy", "mean"),
        f1=("f1", "mean"),
        retrains=("retrained", "sum"),
        cost=("cost_seconds", "sum")
    ).reset_index()
    summary = per_seed.groupby(["policy", "policy_name"]).agg(
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        retrains_mean=("retrains", "mean"),
        retrains_std=("retrains", "std"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std")
    ).round(4).reset_index()

    content.append("## 3. Intrusion Stream (results/experiment_results_intrusion.csv)\n")
    content.append(summary.to_markdown(index=False))
    content.append("\n")

# 4. Dataset: Electricity (experiment_results_electricity.csv)
p4 = Path("results/experiment_results_electricity.csv")
if p4.exists():
    df = pd.read_csv(p4)
    eval_df = df.dropna(subset=["accuracy"])
    per_seed = eval_df.groupby(["policy", "policy_name", "seed"]).agg(
        acc=("accuracy", "mean"),
        f1=("f1", "mean"),
        retrains=("retrained", "sum"),
        cost=("cost_seconds", "sum")
    ).reset_index()
    summary = per_seed.groupby(["policy", "policy_name"]).agg(
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        retrains_mean=("retrains", "mean"),
        retrains_std=("retrains", "std"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std")
    ).round(4).reset_index()

    content.append("## 4. Electricity Dataset (results/experiment_results_electricity.csv)\n")
    content.append(summary.to_markdown(index=False))
    content.append("\n")

# 5. Ablation Study (ablation_results.csv)
p5 = Path("results/ablation_results.csv")
if p5.exists():
    df = pd.read_csv(p5)
    eval_df = df.dropna(subset=["accuracy"])
    per_seed = eval_df.groupby(["ablation", "seed"]).agg(
        acc=("accuracy", "mean"),
        f1=("f1", "mean"),
        retrains=("retrained", "sum"),
        cost=("cost_seconds", "sum")
    ).reset_index()
    summary = per_seed.groupby("ablation").agg(
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        retrains_mean=("retrains", "mean"),
        retrains_std=("retrains", "std"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std")
    ).round(4).reset_index()

    content.append("## 5. Ablation Study — stat_only vs. full_dis (results/ablation_results.csv)\n")
    content.append(summary.to_markdown(index=False))
    content.append("\n")

# 6. Null Stream Evaluations
p6 = Path("results/null_stream_summary.csv")
if p6.exists():
    df = pd.read_csv(p6)
    content.append("## 6. Null Stream Evaluation — Independent Features (results/null_stream_summary.csv)\n")
    content.append(df.to_markdown(index=False))
    content.append("\n")

p6_corr = Path("results/null_stream_correlated_summary.csv")
if p6_corr.exists():
    df_corr = pd.read_csv(p6_corr)
    content.append("## 6b. Null Stream Evaluation — Correlated Features (results/null_stream_correlated_summary.csv)\n")
    content.append(df_corr.to_markdown(index=False))
    content.append("\n")

p6_elec = Path("results/null_stream_elec2_null_summary.csv")
if p6_elec.exists():
    df_elec = pd.read_csv(p6_elec)
    content.append("## 6c. Null Stream Evaluation — ELEC2 Stationary Slice (results/null_stream_elec2_null_summary.csv)\n")
    content.append(df_elec.to_markdown(index=False))
    content.append("\n")

# 7. Changepoint Detection
cp_path = Path("results/changepoint_summary.csv")
if cp_path.exists():
    df = pd.read_csv(cp_path)
    content.append("## 7. Known Changepoint Detection — Mixed Covariate+Concept Drift (results/changepoint_summary.csv)\n")
    content.append(df.to_markdown(index=False))
    content.append("\n")

# 7b. Concept-Only Changepoint Detection
cp_concept_path = Path("results/changepoint_concept_only_summary.csv")
if cp_concept_path.exists():
    df_concept = pd.read_csv(cp_concept_path)
    content.append("## 7b. Known Changepoint Detection — Pure Concept Drift (results/changepoint_concept_only_summary.csv)\n")
    content.append(df_concept.to_markdown(index=False))
    content.append("\n")

# 8. Policy Operating-Point Curves
p8 = Path("results/policy_curves_summary.csv")
if p8.exists():
    df_curves = pd.read_csv(p8)
    content.append("## 8. Policy Operating-Point Curves & Pareto Front (results/policy_curves_summary.csv)\n")
    content.append(df_curves.to_markdown(index=False))
    content.append("\n> **Visual Plot:** Saved to `results/policy_curves.png`\n\n")

# 9. Sensitivity Grid
p9 = Path("results/sensitivity_results.csv")
if p9.exists():
    df_sens = pd.read_csv(p9)
    eval_df = df_sens.dropna(subset=["accuracy"])
    per_seed = eval_df.groupby(["lambda", "lookback", "seed"]).agg(
        acc=("accuracy", "mean"),
        f1=("f1", "mean"),
        retrains=("retrained", "sum")
    ).reset_index()
    sens_sum = per_seed.groupby(["lambda", "lookback"]).agg(
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        retrains_mean=("retrains", "mean"),
        retrains_std=("retrains", "std")
    ).round(4).reset_index()

    content.append("## 9. Sensitivity Analysis on Holdout (results/sensitivity_results.csv)\n")
    content.append(sens_sum.to_markdown(index=False))
    content.append("\n")

output_path = Path("planning/results_summary.md")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(content), encoding="utf-8")
print("Wrote planning/results_summary.md successfully.")
