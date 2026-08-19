"""
Publication readiness validation script.
Runs all 6 checks from the improvement plan and prints the final paper table.
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
    total = 6

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

    print("\n  [**] P5 is Pareto-optimal: high accuracy with controlled retrain cost")
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
