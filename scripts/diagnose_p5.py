"""Diagnose whether P5 (DIS-Fused) has genuinely different behavior from P1 (Fixed-Schedule).

Two hard checks:
  1. Window-level retrain overlap: if P5's retrain set == P1's for every seed, P5 collapsed.
  2. Pareto dominance: computed from data, not asserted.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "results" / "experiment_results.csv"

df = pd.read_csv(CSV_PATH)
seeds = sorted(df["seed"].unique())
last_w = df["window_id"].max()

print("=" * 60)
print("CHECK 1: P5 vs P1 retrain window overlap (per seed)")
print("=" * 60)

all_same = True
for seed in seeds:
    p1_wins = set(df[(df["policy"] == "p1_fixed") & (df["seed"] == seed) & df["retrained"]]["window_id"].tolist())
    p5_wins = set(df[(df["policy"] == "p5_dis_fused") & (df["seed"] == seed) & df["retrained"]]["window_id"].tolist())
    same = p1_wins == p5_wins
    only_p1 = sorted(p1_wins - p5_wins)
    only_p5 = sorted(p5_wins - p1_wins)
    if not same:
        all_same = False
    status = "IDENTICAL" if same else "DIFFERENT"
    print(f"  seed {seed}: [{status}]")
    print(f"    P1 windows: {sorted(p1_wins)}")
    print(f"    P5 windows: {sorted(p5_wins)}")
    if only_p1:
        print(f"    Only P1: {only_p1}")
    if only_p5:
        print(f"    Only P5: {only_p5}")

print()
if all_same:
    print("[FAIL] P5 retrains on IDENTICAL windows to P1 on EVERY seed.")
    print("       P5 has mechanistically collapsed into P1 — the adaptive")
    print("       mechanism is not driving decisions. Result is NOT valid.")
else:
    n_diff = sum(
        set(df[(df["policy"] == "p1_fixed") & (df["seed"] == s) & df["retrained"]]["window_id"].tolist()) !=
        set(df[(df["policy"] == "p5_dis_fused") & (df["seed"] == s) & df["retrained"]]["window_id"].tolist())
        for s in seeds
    )
    print(f"[PASS] P5 and P1 differ on {n_diff}/{len(seeds)} seeds — adaptive mechanism is active.")

print()
print("=" * 60)
print("CHECK 2: ATC threshold — did it ever exit warmup?")
print("=" * 60)

p5_all = df[df["policy"] == "p5_dis_fused"].copy()
warmup_threshold = 0.1
always_warmup = (p5_all["threshold"] == warmup_threshold).all()
max_threshold = p5_all["threshold"].max()
n_above_warmup = (p5_all["threshold"] > warmup_threshold).sum()

print(f"  warmup_threshold = {warmup_threshold}")
print(f"  max observed threshold = {max_threshold:.6f}")
print(f"  windows where threshold > warmup_threshold: {n_above_warmup} / {len(p5_all)}")

if always_warmup:
    print("  [FAIL] ATC threshold NEVER exceeded warmup value.")
    print("         Every retrain decision was driven by the warmup threshold,")
    print("         not the adaptive theta(t) = mu + lambda*sigma formula.")
    print("         The 'adaptive' part of DIS-Fused did not function.")
else:
    print(f"  [PASS] ATC threshold exceeded warmup value on {n_above_warmup} windows.")

print()
print("=" * 60)
print("CHECK 3: DIS signal — is it providing discriminating information?")
print("=" * 60)

dis_nonzero = p5_all[p5_all["window_id"] > 0]["dis"]
n_zero = (dis_nonzero == 0.0).sum()
n_total = len(dis_nonzero)
pct_zero = 100 * n_zero / n_total
print(f"  DIS=0.0 on {n_zero}/{n_total} windows ({pct_zero:.1f}%) after window 0")
print(f"  DIS>0.0 on {n_total - n_zero}/{n_total} windows ({100-pct_zero:.1f}%)")
if pct_zero > 50:
    print("  [WARN] Majority of windows have DIS=0.0 — normalizer warmup after each")
    print("         retrain is suppressing the signal for most of the run.")

print()
print("=" * 60)
print("CHECK 4: Pareto dominance (computed from data)")
print("=" * 60)

per_seed = df.groupby(["policy", "seed"]).agg(
    acc=("accuracy", "mean"),
    cost=("cumulative_cost", "last"),
    retrains=("cumulative_retrains", "last"),
).reset_index()

summary = per_seed.groupby("policy").agg(
    mean_acc=("acc", "mean"),
    mean_cost=("cost", "mean"),
    mean_retrains=("retrains", "mean"),
).reset_index()

print("  Per-policy mean accuracy vs mean cumulative cost:")
print(f"  {'Policy':30s} {'Acc':>8s} {'Cost(s)':>9s} {'Retrains':>9s}")
print("  " + "-" * 60)
for _, row in summary.sort_values("mean_acc", ascending=False).iterrows():
    print(f"  {row['policy']:30s} {row['mean_acc']:8.4f} {row['mean_cost']:9.3f} {row['mean_retrains']:9.1f}")

# Strict Pareto dominance: policy A dominates policy B if A has higher acc AND lower or equal cost
print()
p5_acc = summary[summary["policy"] == "p5_dis_fused"]["mean_acc"].values[0]
p5_cost = summary[summary["policy"] == "p5_dis_fused"]["mean_cost"].values[0]
p1_acc = summary[summary["policy"] == "p1_fixed"]["mean_acc"].values[0]
p1_cost = summary[summary["policy"] == "p1_fixed"]["mean_cost"].values[0]

print(f"  P5 vs P1: acc_diff={p5_acc - p1_acc:+.6f}, cost_diff={p5_cost - p1_cost:+.4f}")

dominated_by = []
for _, row in summary.iterrows():
    other = row["policy"]
    if other == "p5_dis_fused":
        continue
    if row["mean_acc"] >= p5_acc and row["mean_cost"] <= p5_cost:
        dominated_by.append(other)

if dominated_by:
    print(f"  [FAIL] P5 is Pareto-DOMINATED by: {dominated_by}")
    print("         'Pareto-optimal' claim in paper table is FALSE.")
else:
    print("  [PASS] P5 is not strictly dominated by any other policy.")
    print("         (Note: if P5==P1 on both axes, this check is vacuous.)")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
collapse = all_same
atc_stuck = always_warmup
print(f"  P5 == P1 windows on all seeds: {collapse}")
print(f"  ATC never exited warmup:       {atc_stuck}")
if collapse and atc_stuck:
    print()
    print("  CONCLUSION: P5's 'adaptive' mechanism did not function.")
    print("  DIS-Fused silently became Fixed-Schedule (every 3 windows).")
    print("  The claimed result is NOT valid without fixing the ATC warmup saturation.")
    sys.exit(1)
else:
    print()
    print("  CONCLUSION: P5 shows genuinely distinct behavior from P1.")
    sys.exit(0)
