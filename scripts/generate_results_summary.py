import pandas as pd
from pathlib import Path

content = []
content.append("# DRIFT-X Experiment Results Summary\n")

# 1. Main Benchmark (experiment_results.csv)
df = pd.read_csv("results/experiment_results.csv")
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

content.append("## 1. Main Benchmark (results/experiment_results.csv)\n")
content.append(summary.to_markdown(index=False))
content.append("\n")

# 2. Dataset: Fraud (experiment_results_fraud.csv)
df = pd.read_csv("results/experiment_results_fraud.csv")
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
df = pd.read_csv("results/experiment_results_intrusion.csv")
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
if Path("results/experiment_results_electricity.csv").exists():
    df = pd.read_csv("results/experiment_results_electricity.csv")
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
df = pd.read_csv("results/ablation_results.csv")
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

content.append("## 5. Ablation Study (results/ablation_results.csv)\n")
content.append(summary.to_markdown(index=False))
content.append("\n")

# 6. Null Stream (null_stream_summary.csv)
df = pd.read_csv("results/null_stream_summary.csv")
content.append("## 6. Null Stream Evaluation (results/null_stream_summary.csv)\n")
content.append(df.to_markdown(index=False))
content.append("\n")

# 7. Changepoint Detection (changepoint_summary.csv)
df = pd.read_csv("results/changepoint_summary.csv")
content.append("## 7. Known Changepoint Detection (results/changepoint_summary.csv)\n")
content.append(df.to_markdown(index=False))
content.append("\n")

# 8. Policy Operating-Point Curves (policy_curves_summary.csv)
df = pd.read_csv("results/policy_curves_summary.csv")
content.append("## 8. Policy Operating-Point Curves (results/policy_curves_summary.csv)\n")
content.append(df.to_markdown(index=False))
content.append("\n")

# 9. Sensitivity Grid (sensitivity_results.csv)
df_sens = pd.read_csv("results/sensitivity_results.csv")
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
output_path.write_text("\n".join(content), encoding="utf-8")
print("Wrote planning/results_summary.md successfully.")
