"""
Publication readiness validation script.
Runs all 8 checks and prints the final paper table.

Checks 1-6: aggregate-level publication readiness (original).
Check 7:    window-level retrain overlap — rules out P5 silently collapsing into any baseline.
Check 8:    Pareto dominance using retrain_count (primary cost), not wall-clock seconds.

Guards:
  - Every input CSV must contain a 'data_source' column (provenance).
  - Window 0 must have NaN accuracy in every input CSV (test-then-train protocol).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_and_guard(path: Path, name: str) -> pd.DataFrame:
    """Load a results CSV and run provenance + protocol guards."""
    df = pd.read_csv(path)

    # Guard 1: data_source column must be present
    if "data_source" not in df.columns:
        raise ValueError(
            f"[GUARD FAIL] '{name}' is missing the 'data_source' column. "
            "Re-run the experiment with the fixed runner (data_source_type in config)."
        )
    print(f"  [GUARD OK] {name}: data_source present — "
          f"{sorted(df['data_source'].unique())}")

    # Guard 2: window 0 accuracy must be NaN in all rows (test-then-train protocol)
    w0 = df[df["window_id"] == 0]
    if not w0.empty and not w0["accuracy"].isna().all():
        non_nan_policies = w0[w0["accuracy"].notna()]["policy"].unique().tolist()
        raise ValueError(
            f"[GUARD FAIL] '{name}' has non-NaN accuracy at window 0 for "
            f"policies: {non_nan_policies}. "
            "This indicates in-sample evaluation was logged — re-run the experiment."
        )
    print(f"  [GUARD OK] {name}: window-0 accuracy is NaN (test-then-train protocol confirmed)")

    return df


def main():
    print("=" * 60)
    print("GUARD CHECKS (must pass before validation begins)")
    print("=" * 60)

    df = _load_and_guard(PROJECT_ROOT / "results" / "experiment_results.csv",
                         "experiment_results.csv")
    abl = _load_and_guard(PROJECT_ROOT / "results" / "ablation_results.csv",
                          "ablation_results.csv")
    sens = pd.read_csv(PROJECT_ROOT / "results" / "sensitivity_results.csv")
    # Sensitivity CSV does not need the guards (it's p5 only, all windows >0)

    print()
    print("=" * 60)
    print("PUBLICATION READINESS VALIDATION")
    print("=" * 60)

    passed = 0
    total = 8

    # CHECK 1: Seeds produce variance
    print("\nCHECK 1: Seed variance (std > 0.001 per policy)")
    # Exclude window 0 (NaN) from per-seed mean
    df_eval = df[df["window_id"] > 0]
    per_seed = df_eval.groupby(["policy", "seed"])["accuracy"].mean()
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
    p0_acc = df_eval[df_eval["policy"] == "p0_never"]["accuracy"].mean()
    p5_acc = df_eval[df_eval["policy"] == "p5_dis_fused"]["accuracy"].mean()
    diff = abs(p5_acc - p0_acc)
    ok = diff > 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] P0={p0_acc:.4f}, P5={p5_acc:.4f}, gap={diff:.4f}")
    if ok:
        passed += 1

    # CHECK 5: Ablation variation
    print("\nCHECK 5: Ablation configs produce varied accuracy")
    abl_eval = abl[abl["window_id"] > 0]
    abl_means = abl_eval.groupby("ablation")["accuracy"].mean()
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
        p1_wins = set(df[(df["policy"] == "p1_fixed") & (df["seed"] == seeds[0]) & df["retrained"]]["window_id"].tolist())
        p5_wins = p5_windows_per_seed[seeds[0]]
        print(f"         (seed {seeds[0]} example - P1: {sorted(p1_wins)}, P5: {sorted(p5_wins)})")
        passed += 1

    # CHECK 8: Pareto dominance — uses retrain_count (primary cost, timing-noise-free)
    print("\nCHECK 8: Pareto position of P5 (retrain_count as primary cost)")
    # Use retrain_count - 1 (subtract the mandatory window-0 initial train) averaged per seed.
    # retrain_count at any row is the cumulative count; we take the last value per seed.
    if "retrain_count" in df.columns:
        cost_col = "retrain_count"
        cost_label = "retrains (excl. initial)"
        per_seed_agg = df.groupby(["policy", "seed"]).agg(
            acc=("accuracy", "mean"),          # nanmean implicitly via mean() on floats
            cost=(cost_col, "last"),
        ).reset_index()
        # Subtract 1 to exclude the mandatory window-0 initial training
        per_seed_agg["cost"] = per_seed_agg["cost"] - 1
    else:
        # Fallback to wall-clock if retrain_count missing (old CSV)
        cost_col = "cumulative_cost"
        cost_label = "wall-clock seconds"
        per_seed_agg = df.groupby(["policy", "seed"]).agg(
            acc=("accuracy", "mean"),
            cost=(cost_col, "last"),
        ).reset_index()
        print("  [WARN] retrain_count column not found; falling back to cumulative_cost")

    policy_means = per_seed_agg.groupby("policy").agg(
        mean_acc=("acc", "mean"),
        mean_cost=("cost", "mean"),
    ).reset_index()
    p5_row = policy_means[policy_means["policy"] == "p5_dis_fused"].iloc[0]
    p5_acc_p, p5_cost_p = p5_row["mean_acc"], p5_row["mean_cost"]
    dominated_by = [
        row["policy"] for _, row in policy_means.iterrows()
        if row["policy"] != "p5_dis_fused"
        and row["mean_acc"] >= p5_acc_p
        and row["mean_cost"] <= p5_cost_p
    ]
    print(f"  P5: acc={p5_acc_p:.4f}, {cost_label}={p5_cost_p:.1f}")
    for _, row in policy_means.sort_values("mean_acc", ascending=False).iterrows():
        marker = " <- P5" if row["policy"] == "p5_dis_fused" else ""
        dom = " [dominates P5]" if row["policy"] in dominated_by else ""
        print(f"    {row['policy']:28s} acc={row['mean_acc']:.4f} {cost_label}={row['mean_cost']:.1f}{marker}{dom}")
    if dominated_by:
        print(f"  [FAIL] P5 is Pareto-dominated by: {dominated_by}")
    else:
        print(f"  [PASS] P5 not dominated on (accuracy, {cost_label}) axes simultaneously.")
        passed += 1

    # FINAL VERDICT
    print()
    print("=" * 60)
    if passed == total:
        print(f"RESULT: {passed}/{total} PASSED -- all sanity checks passed")
    else:
        print(f"RESULT: {passed}/{total} PASSED -- {total - passed} issue(s) remain")
    print("=" * 60)

    # PAPER TABLE (exclude window 0 from accuracy averages)
    print("\nPAPER TABLE (mean +/- std across seeds, excluding window-0 NaN):")
    print("-" * 82)
    print(f"  {'Policy':<32} {'Acc':>8} {'+-':>6} {'F1':>8} {'+-':>6} {'Retrains-1':>12}")
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
        sub = df_eval[df_eval["policy"] == policy]
        acc_mean = sub["accuracy"].mean()
        acc_std = sub.groupby("seed")["accuracy"].mean().std()
        f1_mean = sub["f1"].mean()
        f1_std = sub.groupby("seed")["f1"].mean().std()
        # Retrains - 1: subtract mandatory initial train
        if "retrain_count" in df.columns:
            retrains = df[df["policy"] == policy].groupby("seed")["retrain_count"].last().mean() - 1
        else:
            retrains = df[df["policy"] == policy]["cumulative_retrains"].max() - 1
        label = policy_labels.get(policy, policy)
        print(
            f"  {label:<35} {acc_mean:>6.4f} {acc_std:>6.4f}"
            f" {f1_mean:>6.4f} {f1_std:>6.4f} {retrains:>10.1f}"
        )

    # Pareto footnote using retrain_count
    p5r = policy_means[policy_means["policy"] == "p5_dis_fused"].iloc[0]
    p1r = policy_means[policy_means["policy"] == "p1_fixed"].iloc[0]
    acc_diff = p5r["mean_acc"] - p1r["mean_acc"]
    cost_diff = p5r["mean_cost"] - p1r["mean_cost"]
    verdict = (
        "more accurate and fewer retrains" if acc_diff > 0 and cost_diff < 0
        else "trades accuracy for fewer retrains" if acc_diff < 0 and cost_diff < 0
        else "same accuracy, different retrain count"
    )
    print(f"  [**] P5 vs P1: acc={acc_diff:+.4f}, retrains-excl-init={cost_diff:+.1f} -- {verdict}")
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
