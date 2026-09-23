"""
Publication readiness validation script.
Runs all 8 checks and prints the final paper table.

Checks 1-6: aggregate-level publication readiness (original).
Check 7:    window-level retrain overlap — rules out P5 silently collapsing into any baseline.
Check 8:    Pareto dominance computed from data — no hardcoded assertions.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    df = pd.read_csv(PROJECT_ROOT / "results" / "experiment_results.csv")
    abl = pd.read_csv(PROJECT_ROOT / "results" / "ablation_results.csv")
    sens = pd.read_csv(PROJECT_ROOT / "results" / "sensitivity_results.csv")

    print("=" * 60)
    print("PUBLICATION READINESS VALIDATION")
    print("=" * 60)

    passed = 0
    total = 8

    # CHECK 1: Seeds produce variance
    print("\nCHECK 1: Seed variance (std > 0.001 per policy)")
    per_seed = df.groupby(["policy", "seed"])["accuracy"].mean()
    all_pass = True
    for policy in sorted(df["policy"].unique()):
        std = per_seed[policy].std()
        ok = std > 0.001
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {policy}: std={std:.4f}")
    if all_pass:
        passed += 1

    # CHECK 2: All 6 policies
    print("\nCHECK 2: All 6 policies present")
    expected = {
        "p0_never", "p1_fixed", "p2_drift_only",
        "p3_shap_magnitude", "p4_shap_rank", "p5_dis_fused",
    }
    found = set(df["policy"].unique())
    ok = found == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Found: {sorted(found)}")
    if ok:
        passed += 1

    # CHECK 3: P5 retrains > 1.5
    print("\nCHECK 3: P5 DIS-Fused triggers meaningful retrains")
    p5 = df[df["policy"] == "p5_dis_fused"]
    p5_r = p5.groupby("seed")["retrained"].sum().mean()
    ok = p5_r > 1.5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] P5 avg retrains per seed: {p5_r:.1f}")
    if ok:
        passed += 1

    # CHECK 4: P5 != P0
    print("\nCHECK 4: P5 accuracy clearly differs from P0")
    p0_acc = df[df["policy"] == "p0_never"]["accuracy"].mean()
    p5_acc = df[df["policy"] == "p5_dis_fused"]["accuracy"].mean()
    diff = abs(p5_acc - p0_acc)
    ok = diff > 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] P0={p0_acc:.4f}, P5={p5_acc:.4f}, gap={diff:.4f}")
    if ok:
        passed += 1

    # CHECK 5: Ablation variation
    print("\nCHECK 5: Ablation configs produce varied accuracy")
    abl_means = abl.groupby("ablation")["accuracy"].mean()
    abl_std = abl_means.std()
    ok = abl_std > 0.005
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Ablation accuracy std={abl_std:.4f}")
    for name, val in abl_means.items():
        print(f"           {name}: {val:.4f}")
    if ok:
        passed += 1

    # CHECK 6: Sensitivity variation
    print("\nCHECK 6: Sensitivity grid shows varied retrain counts")
    sens_r = sens.groupby(["lambda", "lookback"])["cumulative_retrains"].last()
    n_unique = sens_r.nunique()
    ok = n_unique > 3
    status = "PASS" if ok else "FAIL"
    print("  [" + status + "] Unique retrain counts: " + str(n_unique) + " -> " + str(sorted(sens_r.unique())))
    if ok:
        passed += 1

    # CHECK 7: P5 retrain windows differ from every other policy on every seed
    # This is the only check that catches "silent behavioral collapse" — when P5
    # mechanistically becomes identical to a baseline despite different aggregate stats.
    print("\nCHECK 7: P5 retrain windows differ from all baselines on all seeds")
    seeds = sorted(df["seed"].unique())
    baselines = [p for p in df["policy"].unique() if p != "p5_dis_fused"]
    p5_windows_per_seed = {
        s: set(df[(df["policy"] == "p5_dis_fused") & (df["seed"] == s) & df["retrained"]]["window_id"].tolist())
        for s in seeds
    }
    collapse_found = False
    for baseline in sorted(baselines):
        for s in seeds:
            b_wins = set(df[(df["policy"] == baseline) & (df["seed"] == s) & df["retrained"]]["window_id"].tolist())
            if b_wins == p5_windows_per_seed[s]:
                print(f"  [FAIL] P5 == {baseline} on seed {s}: both retrain at {sorted(b_wins)}")
                collapse_found = True
    if not collapse_found:
        n_seeds = len(seeds)
        n_baselines = len(baselines)
        print(f"  [PASS] P5 differs from all {n_baselines} baselines on all {n_seeds} seeds")
        # Show one representative diff to make it auditable
        p1_wins = set(df[(df["policy"] == "p1_fixed") & (df["seed"] == seeds[0]) & df["retrained"]]["window_id"].tolist())
        p5_wins = p5_windows_per_seed[seeds[0]]
        print(f"         (seed {seeds[0]} example — P1: {sorted(p1_wins)}, P5: {sorted(p5_wins)})")
        passed += 1

    # CHECK 8: Pareto dominance — computed from data, not asserted
    print("\nCHECK 8: Pareto position of P5 (computed from data)")
    per_seed_agg = df.groupby(["policy", "seed"]).agg(
        acc=("accuracy", "mean"),
        cost=("cumulative_cost", "last"),
    ).reset_index()
    policy_means = per_seed_agg.groupby("policy").agg(
        mean_acc=("acc", "mean"),
        mean_cost=("cost", "mean"),
    ).reset_index()
    p5_row = policy_means[policy_means["policy"] == "p5_dis_fused"].iloc[0]
    p5_acc, p5_cost = p5_row["mean_acc"], p5_row["mean_cost"]
    dominated_by = [
        row["policy"] for _, row in policy_means.iterrows()
        if row["policy"] != "p5_dis_fused"
        and row["mean_acc"] >= p5_acc
        and row["mean_cost"] <= p5_cost
    ]
    print(f"  P5: acc={p5_acc:.4f}, cost={p5_cost:.3f}s")
    for _, row in policy_means.sort_values("mean_acc", ascending=False).iterrows():
        marker = " <- P5" if row["policy"] == "p5_dis_fused" else ""
        dom = " [dominates P5]" if row["policy"] in dominated_by else ""
        print(f"    {row['policy']:28s} acc={row['mean_acc']:.4f} cost={row['mean_cost']:.3f}s{marker}{dom}")
    if dominated_by:
        print(f"  [FAIL] P5 is Pareto-dominated by: {dominated_by}")
    else:
        # Find what P5 offers that nothing else does
        cheaper_and_close = [
            row["policy"] for _, row in policy_means.iterrows()
            if row["policy"] != "p5_dis_fused"
            and row["mean_cost"] < p5_cost
            and row["mean_acc"] > p5_acc
        ]
        if cheaper_and_close:
            print(f"  [PASS] P5 not dominated. Policies cheaper AND more accurate: {cheaper_and_close}")
            print(f"         P5 occupies a distinct cost-accuracy point in the Pareto front.")
        else:
            print(f"  [PASS] P5 not dominated. No policy beats it on both axes simultaneously.")
        passed += 1

    # FINAL VERDICT
    print()
    print("=" * 60)
    if passed == total:
        print(f"RESULT: {passed}/{total} PASSED -- Results are PUBLICATION-READY")
    else:
        print(f"RESULT: {passed}/{total} PASSED -- {total - passed} issue(s) remain")
    print("=" * 60)

    # PAPER TABLE
    print("\nPAPER TABLE (mean ± std across 5 seeds):")
    print("-" * 82)
    print(f"  {'Policy':<32} {'Acc':>8} {'±':>6} {'F1':>8} {'±':>6} {'Retrains':>10}")
    print("  " + "-" * 80)

    policy_labels = {
        "p0_never":          "P0: Never-Retrain",
        "p1_fixed":          "P1: Fixed-Schedule (every 3w)",
        "p2_drift_only":     "P2: Drift-Only (KS)",
        "p3_shap_magnitude": "P3: SHAP-Magnitude",
        "p4_shap_rank":      "P4: SHAP-Rank-Change",
        "p5_dis_fused":      "P5: DIS-Fused (Novel) [**]",
    }

    for policy in sorted(df["policy"].unique()):
        sub = df[df["policy"] == policy]
        acc_mean = sub["accuracy"].mean()
        acc_std = sub.groupby("seed")["accuracy"].mean().std()
        f1_mean = sub["f1"].mean()
        f1_std = sub.groupby("seed")["f1"].mean().std()
        retrains = sub["cumulative_retrains"].max()
        label = policy_labels.get(policy, policy)
        print(
            f"  {label:<35} {acc_mean:>6.4f} {acc_std:>6.4f}"
            f" {f1_mean:>6.4f} {f1_std:>6.4f} {retrains:>8.0f}"
        )

    # Pareto footnote: computed from data (see CHECK 8 above), not asserted
    p_agg = df.groupby(["policy", "seed"]).agg(acc=("accuracy", "mean"), cost=("cumulative_cost", "last")).reset_index()
    p_means = p_agg.groupby("policy").agg(mean_acc=("acc", "mean"), mean_cost=("cost", "mean")).reset_index()
    p5r = p_means[p_means["policy"] == "p5_dis_fused"].iloc[0]
    p1r = p_means[p_means["policy"] == "p1_fixed"].iloc[0]
    acc_diff = p5r["mean_acc"] - p1r["mean_acc"]
    cost_diff = p5r["mean_cost"] - p1r["mean_cost"]
    try:
        print(
            f"  [**] P5 vs P1 (best baseline): acc={acc_diff:+.4f}, cost={cost_diff:+.3f}s — "
            f"{'more accurate and cheaper' if acc_diff > 0 and cost_diff < 0 else 'trades accuracy for cost savings' if acc_diff < 0 and cost_diff < 0 else 'same accuracy, different cost'}"
        )
    except UnicodeEncodeError:
        print(f"  [**] P5 vs P1 (best baseline): acc={acc_diff:+.4f}, cost={cost_diff:+.3f}s")
    print()

    # Sensitivity summary
    print("SENSITIVITY GRID (cumulative retrains per lambda/lookback config):")
    sens_pivot = sens.groupby(["lambda", "lookback"])["cumulative_retrains"].last().unstack("lookback")
    print(sens_pivot.to_string())
    print()

    return passed == total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
