"""DRIFT-X Comprehensive Results Summary Generator.

Inspects all result files, verifies provenance (git commit and config hash),
flags stale runs, and generates planning/results_summary.md.
"""
from pathlib import Path
import hashlib
import subprocess
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ResultsSummary")


def frozen_hash() -> str:
    path = PROJECT_ROOT / "configs" / "frozen.yaml"
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return "unknown"


FROZEN_HASH = frozen_hash()

try:
    CURRENT_COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT
    ).decode().strip()
except Exception:
    CURRENT_COMMIT = "unknown"


def check_file_provenance(path: Path) -> Dict[str, Any]:
    """Check whether a CSV file matches current commit and frozen config."""
    if not path.exists():
        return {"exists": False, "status": "MISSING", "hash": "N/A", "commit": "N/A"}
    try:
        df_head = pd.read_csv(path, nrows=50)
        h_vals = df_head["frozen_hash"].dropna().unique() if "frozen_hash" in df_head.columns else (
            df_head["config_hash"].dropna().unique() if "config_hash" in df_head.columns else []
        )
        c_vals = df_head["git_commit"].dropna().unique() if "git_commit" in df_head.columns else []
        h = str(h_vals[0]) if len(h_vals) > 0 else "none"
        c = str(c_vals[0]) if len(c_vals) > 0 else "none"

        fh = df_head["frozen_hash"].dropna().unique() if "frozen_hash" in df_head.columns else []
        dirty = df_head["git_dirty"].dropna().unique() if "git_dirty" in df_head.columns else []

        # Check if commit is clean and matches current tree or is an ancestor in repo
        commit_valid = False
        if c != "none":
            if c.startswith(CURRENT_COMMIT) or CURRENT_COMMIT.startswith(c):
                commit_valid = True
            else:
                try:
                    res = subprocess.run(
                        ["git", "merge-base", "--is-ancestor", c, "HEAD"],
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                    )
                    commit_valid = (res.returncode == 0)
                except Exception:
                    commit_valid = False

        ok = (
            commit_valid
            and len(fh) > 0
            and str(fh[0]) == FROZEN_HASH
            and len(dirty) > 0
            and not bool(dirty[0])
        )
        status = "CURRENT" if ok else "UNVERIFIED"
        return {"exists": True, "status": status, "hash": h, "commit": c}
    except Exception as e:
        return {"exists": True, "status": "ERROR", "hash": str(e), "commit": "error"}


def findings(df_bt: pd.DataFrame, df_bm: pd.DataFrame) -> str:
    """Generate Section 9 findings dynamically from statistical tests and metrics data."""
    out = []
    if df_bt is not None and not df_bt.empty:
        for other in ["p2_drift_only", "p7_weighted_ks", "p5_mag_only", "p3_shap_magnitude"]:
            r = df_bt[(df_bt.vs == other) & (df_bt.metric == "auc")]
            if r.empty:
                continue
            r = r.iloc[0]
            verdict = ("no significant difference" if r.p_holm >= 0.05
                       else ("P5 better" if r.diff_mean > 0 else f"{other} better"))
            out.append(f"- P5 vs {other}, AUC: {r.diff_mean:+.4f} "
                       f"[{r.ci_95_lo:+.4f}, {r.ci_95_hi:+.4f}], Holm p={r.p_holm:.3f}, n={int(r.n)}: {verdict}.")
    if df_bm is not None and not df_bm.empty:
        p5_rows = df_bm[df_bm.policy == "p5_dis_fused"]
        if not p5_rows.empty:
            p5 = p5_rows.iloc[0]
            rec = getattr(p5, "cp_recall", getattr(p5, "mean_cp_recall", 0.0))
            prec = getattr(p5, "cp_precision", getattr(p5, "mean_cp_precision", 0.0))
            ch_rec = getattr(p5, "chance_recall", np.nan)
            ch_prec = getattr(p5, "chance_precision", np.nan)
            if pd.notna(ch_rec) and pd.notna(ch_prec):
                chance_str = f" (chance {ch_rec:.2f}/{ch_prec:.2f})"
            else:
                chance_str = ""
            out.append(f"- P5 recall/precision at ~4 retrains: {rec:.2f}/{prec:.2f}{chance_str}.")
    return "\n".join(out) if out else "- No test or metrics data available yet."


def build_summary() -> str:
    content = []
    content.append("# DRIFT-X Comprehensive Experiment Results Summary\n")
    content.append(f"**Target Frozen Hash:** `{FROZEN_HASH}`  |  **Current Git Commit:** `{CURRENT_COMMIT}`\n\n")

    files_to_check = [
        ("Main Benchmark", Path("results/experiment_results.csv")),
        ("Fraud Stream", Path("results/experiment_results_fraud.csv")),
        ("Intrusion Stream", Path("results/experiment_results_intrusion.csv")),
        ("Electricity Stream", Path("results/experiment_results_electricity.csv")),
        ("Weather Stream (Real)", Path("results/real_weather_budget_curves_summary.csv")),
        ("Decoupled Confirmatory (Seeds 4000-4029)", Path("results/budget_curves_mixed_decoupled_s4000_4029_confirmatory_summary.csv")),
        ("Null Stream (Gaussian)", Path("results/null_stream_results.csv")),
        ("Null Stream (ELEC2 Shuffled)", Path("results/null_stream_elec2_null_results.csv")),
        ("Changepoints (Mixed)", Path("results/changepoint_results.csv")),
        ("Changepoints (Concept-Only)", Path("results/changepoint_concept_only_results.csv")),
        ("Sensitivity Sweep", Path("results/sensitivity_results.csv")),
        ("Budget Curves Raw", Path("results/budget_curves_mixed_raw.csv")),
        ("Budget Curves Metrics", Path("results/budget_curves_mixed_metrics.csv")),
    ]

    prov_rows = []
    for label, p in files_to_check:
        info = check_file_provenance(PROJECT_ROOT / p)
        prov_rows.append({
            "Experiment / Output": label,
            "File": str(p),
            "Status": f"**{info['status']}**" if info["status"] == "CURRENT" else f"`{info['status']}`",
            "Config Hash": info["hash"],
            "Git Commit": info["commit"],
        })

    content.append("## Provenance & Reproducibility Audit\n")
    content.append(pd.DataFrame(prov_rows).to_markdown(index=False))
    content.append("\n\n---\n")

    # 1. Main Benchmark
    p1 = PROJECT_ROOT / "results" / "experiment_results.csv"
    info1 = check_file_provenance(p1)
    tag1 = "" if info1["status"] == "CURRENT" else " [STALE - PRE-FIX]"
    content.append(f"## 1. Main Benchmark — All 10 Policies{tag1}\n")
    if p1.exists():
        df1 = pd.read_csv(p1)
        eval_df = df1.dropna(subset=["accuracy"])
        per_seed = eval_df.groupby(["policy", "policy_name", "seed"]).agg(
            acc=("accuracy", "mean"),
            f1=("f1", "mean"),
            retrains=("retrained", "sum"),
            cost=("cost_seconds", "sum")
        ).reset_index()
        s1 = per_seed.groupby(["policy", "policy_name"]).agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            retrains_mean=("retrains", "mean"),
            retrains_std=("retrains", "std"),
            cost_mean=("cost", "mean"),
            cost_std=("cost", "std")
        ).round(4).reset_index()
        content.append(s1.to_markdown(index=False))
        content.append("\n\n")

    # 2-4. Multi-Dataset
    for ds_id, ds_name, fname in [
        ("2", "Fraud Dataset (Synthetic)", "experiment_results_fraud.csv"),
        ("3", "Intrusion Dataset (Synthetic)", "experiment_results_intrusion.csv"),
        ("4", "Electricity Dataset (Real ELEC2, 7 features, date removed)", "experiment_results_electricity.csv"),
    ]:
        p = PROJECT_ROOT / "results" / fname
        info = check_file_provenance(p)
        tag = "" if info["status"] == "CURRENT" else " [STALE - PRE-FIX]"
        content.append(f"## {ds_id}. {ds_name}{tag}\n")
        if p.exists():
            df = pd.read_csv(p)
            eval_df = df.dropna(subset=["accuracy"])
            per_seed = eval_df.groupby(["policy", "policy_name", "seed"]).agg(
                acc=("accuracy", "mean"),
                retrains=("retrained", "sum"),
                cost=("cost_seconds", "sum")
            ).reset_index()
            s = per_seed.groupby(["policy", "policy_name"]).agg(
                acc_mean=("acc", "mean"),
                acc_std=("acc", "std"),
                retrains_mean=("retrains", "mean"),
                retrains_std=("retrains", "std"),
                cost_mean=("cost", "mean"),
            ).round(4).reset_index()
            content.append(s.to_markdown(index=False))
            content.append("\n\n")

    # 5. Null Streams
    content.append("## 5. Null Stream Calibration (False Positive Rate at Full Scale)\n")
    p_null_ind = PROJECT_ROOT / "results" / "null_stream_summary.csv"
    if p_null_ind.exists():
        content.append("### 5a. Independent Gaussian Null (20 windows × 5 seeds = 95 eval windows)\n")
        content.append(pd.read_csv(p_null_ind).to_markdown(index=False))
        content.append("\n\n")

    p_null_elec = PROJECT_ROOT / "results" / "null_stream_elec2_null_summary.csv"
    if p_null_elec.exists():
        content.append("### 5b. Shuffled Disjoint ELEC2 Null (20 windows × 5 seeds = 95 eval windows, date excluded)\n")
        content.append(pd.read_csv(p_null_elec).to_markdown(index=False))
        content.append("\n\n")

    # 6. Known Changepoint Detection
    p_cp = PROJECT_ROOT / "results" / "changepoint_summary.csv"
    info_cp = check_file_provenance(PROJECT_ROOT / "results" / "changepoint_results.csv")
    tag_cp = "" if info_cp["status"] == "CURRENT" else " [STALE - PRE-FIX]"
    content.append(f"## 6. Known Changepoint Detection — Mixed Covariate+Concept Drift{tag_cp}\n")
    if p_cp.exists():
        content.append(pd.read_csv(p_cp).to_markdown(index=False))
        content.append("\n\n")

    p_cp_c = PROJECT_ROOT / "results" / "changepoint_concept_only_summary.csv"
    if p_cp_c.exists():
        content.append("### 6b. Known Changepoint Detection — Pure Concept Drift (Blindness of P2 Revealed)\n")
        content.append(pd.read_csv(p_cp_c).to_markdown(index=False))
        content.append("\n\n")

    # 7. Sensitivity Grid
    p_sens = PROJECT_ROOT / "results" / "sensitivity_results.csv"
    info_sens = check_file_provenance(p_sens)
    tag_sens = "" if info_sens["status"] == "CURRENT" else " [STALE - PRE-FIX]"
    content.append(f"## 7. Sensitivity Analysis on Holdout Streams (Adaptive Mode post-warmup){tag_sens}\n")
    if p_sens.exists():
        df_sens = pd.read_csv(p_sens)
        eval_df = df_sens.dropna(subset=["accuracy"])
        per_seed = eval_df.groupby(["lambda", "lookback", "seed"]).agg(
            acc=("accuracy", "mean"),
            retrains=("retrained", "sum")
        ).reset_index()
        s_sens = per_seed.groupby(["lambda", "lookback"]).agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            retrains_mean=("retrains", "mean"),
            retrains_std=("retrains", "std")
        ).round(4).reset_index()
        content.append(s_sens.to_markdown(index=False))
        content.append("\n\n")

    # 8. Matched-Budget Multi-Stream Evaluation (30 Streams, seeds 1000-1029)
    p_bm = PROJECT_ROOT / "results" / "budget_curves_mixed_metrics.csv"
    p_bt = PROJECT_ROOT / "results" / "budget_curves_mixed_tests.csv"
    df_bm = pd.DataFrame()
    df_bt = pd.DataFrame()
    df_bm_mean = pd.DataFrame()

    if p_bm.exists() and p_bm.stat().st_size > 0:
        try:
            df_bm = pd.read_csv(p_bm)
            content.append("## 8. Matched-Budget Multi-Stream Benchmark\n")
            content.append(
                "Every curve is dual-anchored at (0, P0 accuracy) and (N_eval, always-retrain accuracy), "
                "guaranteeing identical evaluation across all streams without dropped instances.\n\n"
            )
            agg_dict = {
                "n_streams": ("stream", "count"),
                "acc_at_3": ("acc@3", "mean"),
                "acc_at_4": ("acc@4", "mean"),
                "acc_at_5": ("acc@5", "mean"),
                "curve_auc": ("auc", "mean"),
            }
            if "near@3" in df_bm.columns:
                agg_dict["near_3"] = ("near@3", "mean")
            if "near@4" in df_bm.columns:
                agg_dict["near_4"] = ("near@4", "mean")
            if "near@5" in df_bm.columns:
                agg_dict["near_5"] = ("near@5", "mean")

            # Check for cp_recall or mean_cp_recall
            rec_col = "cp_recall" if "cp_recall" in df_bm.columns else ("mean_cp_recall" if "mean_cp_recall" in df_bm.columns else None)
            prec_col = "cp_precision" if "cp_precision" in df_bm.columns else ("mean_cp_precision" if "mean_cp_precision" in df_bm.columns else None)
            if rec_col:
                agg_dict["cp_recall"] = (rec_col, "mean")
            if prec_col:
                agg_dict["cp_precision"] = (prec_col, "mean")
            if "chance_recall" in df_bm.columns:
                agg_dict["chance_recall"] = ("chance_recall", "mean")
            if "chance_precision" in df_bm.columns:
                agg_dict["chance_precision"] = ("chance_precision", "mean")

            s_bm = df_bm.groupby("policy").agg(**agg_dict).reset_index().sort_values("curve_auc", ascending=False).round(4)
            content.append("### Matched-Budget Performance Summary\n")
            content.append(s_bm.to_markdown(index=False))
            content.append("\n\n")

            df_bm_mean = df_bm.groupby("policy").mean(numeric_only=True).reset_index()
        except Exception as e:
            logger.warning(f"Failed to read {p_bm}: {e}")

    if p_bt.exists() and p_bt.stat().st_size > 0:
        try:
            df_bt = pd.read_csv(p_bt)
            if not df_bt.empty:
                content.append("### Paired Statistical Tests vs P5 DIS-Fused (Wilcoxon Signed-Rank with Holm-Bonferroni Correction & 95% Bootstrap CIs)\n")
                content.append(df_bt.to_markdown(index=False))
                content.append("\n\n")
        except Exception as e:
            logger.warning(f"Failed to read {p_bt}: {e}")

    # 9. Scientific Findings & Explicit Disclosures (dynamically generated)
    content.append("## 9. Scientific Findings & Explicit Disclosures\n\n")
    content.append("### Empirical Findings (Data-Derived)\n")
    content.append(findings(df_bt, df_bm_mean))
    content.append("\n\n")

    content.append("### Label Setting & Supervision Disclosures\n")
    content.append(
        "- **Label-Free Policies (P0, P1, P2, P3, P4, P5, P7, P8, P12 KSWIN):** "
        "Evaluate drift and make retraining decisions entirely on unlabeled features X (or model predicted probabilities for KSWIN), requiring zero immediate ground-truth labels.\n"
        "- **Label-Using Policies (P6, P9, P10 ADWIN, P11 DDM, P13 Page-Hinkley):** "
        "Require immediate ground-truth labels y at window end. P6, ADWIN, DDM, and Page-Hinkley monitor classification errors, and P9 computes SHAP values of log loss with true labels.\n"
    )

    out_file = PROJECT_ROOT / "planning" / "results_summary.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(content), encoding="utf-8")
    logger.info(f"Summary written to {out_file}")
    return str(out_file)


if __name__ == "__main__":
    build_summary()
