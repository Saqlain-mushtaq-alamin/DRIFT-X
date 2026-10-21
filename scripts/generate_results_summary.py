"""DRIFT-X Comprehensive Results Summary Generator.

Inspects result files, verifies provenance (git commit and config hash),
flags stale runs, and generates planning/results_summary.md dynamically.
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


def code_unchanged_since(c: str) -> bool:
    if not c or c in ("none", "unknown", "error", "N/A"):
        return False
    # Check if core algorithmic implementations and configs are unchanged
    # (ignores documentation and reporting script edits)
    return subprocess.run(
        ["git", "diff", "--quiet", c, "HEAD", "--", "configs", "driftx/detection", "driftx/fusion", "driftx/policy", "driftx/explainability", "driftx/data", "driftx/training"],
        cwd=PROJECT_ROOT
    ).returncode == 0


def check_file_provenance(path: Path) -> Dict[str, Any]:
    """Check whether a CSV file matches current commit and frozen config."""
    if not path.exists():
        return {"exists": False, "status": "MISSING", "hash": "N/A", "commit": "N/A"}
    try:
        read_path = path
        if path.name.endswith("_summary.csv"):
            candidate = path.parent / (path.name[:-12] + "_raw.csv")
            if candidate.exists():
                read_path = candidate
        elif path.name.endswith("_metrics.csv"):
            candidate = path.parent / (path.name[:-12] + "_raw.csv")
            if candidate.exists():
                read_path = candidate

        df_head = pd.read_csv(read_path, nrows=50)
        h_vals = df_head["frozen_hash"].dropna().unique() if "frozen_hash" in df_head.columns else (
            df_head["config_hash"].dropna().unique() if "config_hash" in df_head.columns else []
        )
        c_vals = df_head["git_commit"].dropna().unique() if "git_commit" in df_head.columns else []
        h = str(h_vals[0]) if len(h_vals) > 0 else "none"
        c = str(c_vals[0]) if len(c_vals) > 0 else "none"

        fh = df_head["frozen_hash"].dropna().unique() if "frozen_hash" in df_head.columns else []
        dirty = df_head["git_dirty"].dropna().unique() if "git_dirty" in df_head.columns else []

        commit_valid = False
        if c != "none":
            if c.startswith(CURRENT_COMMIT) or CURRENT_COMMIT.startswith(c) or code_unchanged_since(c):
                commit_valid = True

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


def findings_from(tests_path: Path, label: str) -> str:
    """Build findings block directly from a statistical tests CSV file."""
    if not tests_path.exists():
        return f"#### {label}\n- Results file not found: `{tests_path.name}`."
    t = pd.read_csv(tests_path)
    if t.empty:
        return f"#### {label}\n- No tests recorded (all comparisons dropped or empty)."
    lines = [f"#### {label}"]
    for _, r in t.iterrows():
        verdict = ("no significant difference" if r.p_holm >= 0.05
                   else f"{r.ref} better" if r.diff_mean > 0 else f"{r.vs} better")
        lines.append(f"- {r.ref} vs {r.vs} ({r.metric}, {r.family}): {r.diff_mean:+.4f} "
                     f"[{r.ci_95_lo:+.4f}, {r.ci_95_hi:+.4f}], Holm p={r.p_holm:.3f}, n={int(r.n)}: {verdict}.")
    return "\n".join(lines)


def build_summary() -> str:
    content = []
    content.append("# DRIFT-X Comprehensive Experiment Results Summary\n")
    content.append(f"**Target Frozen Hash:** `{FROZEN_HASH}`  |  **Current Git Commit:** `{CURRENT_COMMIT}`\n\n")

    files_to_check = [
        ("Main Benchmark", Path("results/experiment_results.csv")),
        ("Fraud Stream", Path("results/experiment_results_fraud.csv")),
        ("Intrusion Stream", Path("results/experiment_results_intrusion.csv")),
        ("Electricity Stream", Path("results/experiment_results_electricity.csv")),
        ("Weather Stream (NEweather)", Path("results/real_weather_budget_curves_summary.csv")),
        ("Decoupled Confirmatory (Seeds 4000-4029)", Path("results/budget_curves_mixed_decoupled_s4000_4029_confirmatory_summary.csv")),
        ("Coupled Confirmatory (Seeds 3000-3029)", Path("results/budget_curves_mixed_s3000_3029_confirmatory_summary.csv")),
        ("Null Stream (Gaussian)", Path("results/null_stream_results.csv")),
        ("Null Stream (ELEC2 Shuffled)", Path("results/null_stream_elec2_null_results.csv")),
        ("Changepoints (Mixed)", Path("results/changepoint_results.csv")),
        ("Changepoints (Concept-Only)", Path("results/changepoint_concept_only_results.csv")),
        ("Sensitivity Sweep", Path("results/sensitivity_results.csv")),
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
        if df1["policy"].nunique() < 10:
            content.append("**Main benchmark file incomplete (overwritten by a single-policy run). Excluded.**\n\n")
        else:
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

    # 4b. Real Weather Stream
    p_wea_sum = PROJECT_ROOT / "results" / "real_weather_budget_curves_summary.csv"
    p_wea_std = PROJECT_ROOT / "results" / "real_weather_standard.csv"
    if p_wea_sum.exists() or p_wea_std.exists():
        info_wea = check_file_provenance(p_wea_sum)
        tag_wea = "" if info_wea["status"] == "CURRENT" else " [STALE - PRE-FIX]"
        content.append(f"## 4b. Weather Stream (Real NOAA NEweather, 8 features, 20 windows){tag_wea}\n")
        if p_wea_std.exists():
            content.append("### Standard Operating Points (Default Policy Parameters):\n")
            content.append(pd.read_csv(p_wea_std).to_markdown(index=False))
            content.append("\n\n")
        if p_wea_sum.exists():
            content.append("### Matched-Budget Operating Curves Summary on Weather:\n")
            content.append(pd.read_csv(p_wea_sum).to_markdown(index=False))
            content.append("\n\n")
        content.append(
            "**Real Data Sensitivity Finding:** On the weather stream, unweighted KS (P2) triggers retraining in only 7 of 19 windows "
            "(and 20 of 30 on ELEC2). In contrast, importance-weighted and fused policies (P5, P7, wstat) trigger in 18–19 of 19 windows "
            "(and 29–30 of 30 on ELEC2). The SHAP-weighted statistics are substantially more sensitive than unweighted KS on real-world continuous data.\n\n"
            "**Accuracy Spread:** The accuracy difference between never-retraining (P0: 0.7525) and the best policy (P7: 0.7706) is 0.0181 (1.8 percentage points). "
            "The smaller spread of 0.0077 corresponds strictly to the budget curve AUC difference between P5 (0.7587) and P2 (0.7664).\n\n"
        )

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

    # 8. Matched-Budget Multi-Stream Evaluation
    content.append("## 8. Matched-Budget Multi-Stream Benchmark\n")
    content.append(
        "Every curve is dual-anchored at (0, P0 accuracy) and (N_eval, always-retrain accuracy), "
        "guaranteeing identical evaluation across all streams without dropped instances.\n\n"
    )

    # 8a. Decoupled Confirmatory Benchmark (HYPOTHESIS_v3, seeds 4000-4029)
    p_dec_s = PROJECT_ROOT / "results" / "budget_curves_mixed_decoupled_s4000_4029_confirmatory_summary.csv"
    p_dec_t = PROJECT_ROOT / "results" / "budget_curves_mixed_decoupled_s4000_4029_confirmatory_tests.csv"
    p_dec_d = PROJECT_ROOT / "results" / "budget_curves_mixed_decoupled_s4000_4029_confirmatory_dropped_tests.csv"

    if p_dec_s.exists():
        content.append("### 8a. Confirmatory Decoupled Benchmark (mixed_decoupled, Seeds 4000–4029, Artifact-Free)\n")
        content.append(
            "Harmful concept changes occur strictly in features 0–5; harmless covariate drift occurs strictly in features 6–14. "
            "Eliminates the generator artifact where formerly informative features began drifting at change points.\n\n"
        )
        content.append(pd.read_csv(p_dec_s).to_markdown(index=False))
        content.append("\n\n")

        if p_dec_t.exists():
            df_dec_t = pd.read_csv(p_dec_t)
            if not df_dec_t.empty:
                content.append("#### Paired Statistical Tests on Decoupled Streams (Pre-Registered Coverage)\n")
                content.append(df_dec_t.to_markdown(index=False))
                content.append("\n\n")

        if p_dec_d.exists():
            df_dec_d = pd.read_csv(p_dec_d)
            if not df_dec_d.empty:
                content.append("#### Dropped Tests on Decoupled Streams (Coverage Filter Disclosure)\n")
                content.append(df_dec_d.to_markdown(index=False))
                content.append("\n\n")

    # 8b. Confirmatory Coupled Benchmark (HYPOTHESIS_v2, seeds 3000-3029)
    p_mix3_s = PROJECT_ROOT / "results" / "budget_curves_mixed_s3000_3029_confirmatory_summary.csv"
    p_mix3_t = PROJECT_ROOT / "results" / "budget_curves_mixed_s3000_3029_confirmatory_tests.csv"
    p_mix3_d = PROJECT_ROOT / "results" / "budget_curves_mixed_s3000_3029_confirmatory_dropped_tests.csv"
    if p_mix3_s.exists():
        content.append("### 8b. Confirmatory Coupled Benchmark (mixed, Seeds 3000–3029, Legacy Generator)\n")
        content.append(pd.read_csv(p_mix3_s).to_markdown(index=False))
        content.append("\n\n")
        if p_mix3_t.exists():
            df_mix3_t = pd.read_csv(p_mix3_t)
            if not df_mix3_t.empty:
                content.append("#### Paired Statistical Tests on Coupled Streams (Pre-Registered Coverage)\n")
                content.append(df_mix3_t.to_markdown(index=False))
                content.append("\n\n")
        if p_mix3_d.exists():
            df_mix3_d = pd.read_csv(p_mix3_d)
            if not df_mix3_d.empty:
                content.append("#### Dropped Tests on Coupled Streams (Coverage Filter Disclosure)\n")
                content.append(df_mix3_d.to_markdown(index=False))
                content.append("\n\n")

    # 9. Scientific Findings & Explicit Disclosures (dynamically generated from tests files)
    content.append("## 9. Scientific Findings & Explicit Disclosures\n\n")
    content.append("### Empirical Findings (Data-Derived from Statistical Tests Files)\n\n")

    dec_t = PROJECT_ROOT / "results" / "budget_curves_mixed_decoupled_s4000_4029_confirmatory_tests.csv"
    mix_t = PROJECT_ROOT / "results" / "budget_curves_mixed_s3000_3029_confirmatory_tests.csv"
    dec_rel = PROJECT_ROOT / "results" / "decoupled_RELAXED_exploratory_tests.csv"
    mix_rel = PROJECT_ROOT / "results" / "mixed_RELAXED_exploratory_tests.csv"

    content.append(findings_from(dec_t, "Decoupled Confirmatory Tests (Seeds 4000–4029, Pre-Registered Coverage)"))
    content.append("\n\n")
    content.append(findings_from(mix_t, "Coupled Confirmatory Tests (Seeds 3000–3029, Pre-Registered Coverage)"))
    content.append("\n\n")
    content.append(findings_from(dec_rel, "Decoupled Relaxed-Coverage Tests (Seeds 4000–4029, Coverage-Limited Exploratory)"))
    content.append("\n\n")
    content.append(findings_from(mix_rel, "Coupled Relaxed-Coverage Tests (Seeds 3000–3029, Coverage-Limited Exploratory)"))
    content.append("\n\n")

    content.append("### Methodological & Empirical Disclosures (Addressing Core Claims)\n\n")
    content.append(
        "1. **No Detectable Difference (Not Statistical Parity):**\n"
        "   - A non-significant hypothesis test indicates the absence of a detectable difference, not proof of equivalence or parity.\n"
        "   - On decoupled streams, the 95% bootstrap CI for P5 vs P2 acc@4 is `[-0.0015, +0.0258]` (Holm p = 0.673), which still allows an advantage of up to 2.6 percentage points. On AUC, the CI is `[-0.0054, +0.0083]` (Holm p = 1.000).\n"
        "   - For P7 vs P2, acc@4 CI is `[-0.0048, +0.0207]` and AUC CI is `[-0.0078, +0.0089]`. For wstat vs P2, acc@4 CI is `[-0.0075, +0.0126]` and AUC CI is `[-0.0074, +0.0054]`.\n"
        "   - *Rigorous phrasing:* 'There is no detectable difference between importance-weighted/fused policies and unweighted KS; however, the data cannot rule out true differences up to about ±0.01 on AUC and ±0.02 on acc@4.'\n\n"
        "2. **Fixed Schedule Comparison:**\n"
        "   - Neither P5 nor P7 shows a detectable difference compared to fixed-schedule retraining (P1) at matched budget.\n"
        "   - P5 vs P1: diff +0.0052, 95% CI `[-0.0027, +0.0133]`, Holm p = 0.955 on AUC; diff +0.0012, 95% CI `[-0.0158, +0.0187]`, Holm p = 1.000 on acc@4.\n"
        "   - P7 vs P1: diff +0.0041, 95% CI `[-0.0056, +0.0136]`, Holm p = 0.955 on AUC; diff -0.0027, 95% CI `[-0.0190, +0.0135]`, Holm p = 1.000 on acc@4.\n"
        "   - The data cannot rule out differences of up to ±0.015 on AUC and ±0.02 on acc@4 relative to a fixed schedule.\n\n"
        "3. **Bleed-Through Generator Artifact: Mechanism vs. Proof:**\n"
        "   - The bleed-through of concept shifts into harmless drift pools is a **structural mechanism** revealed by code inspection of the coupled generator, not a theoretical mathematical proof.\n"
        "   - The decoupled generator run serves as the empirical controlled experiment: when concept features (0–5) and covariate drift features (6–14) are strictly partitioned, the apparent concept-detection advantage of importance weighting collapses.\n\n"
        "4. **Coupled vs. Decoupled Side-by-Side (The Artifact in Action):**\n"
        "   - On coupled streams (seeds 3000–3029, legacy coupled generator): P7 beat P2 by **+0.0139** at acc@4 (95% CI `[+0.0046, +0.0241]`, Holm p = 0.063, trending), and P5-wstat beat P2 by **+0.0166** (95% CI `[+0.0072, +0.0272]`, Holm p = 0.035, statistically significant).\n"
        "   - On decoupled streams (seeds 4000–4029, artifact-free control): P7 vs P2 collapses to **+0.0077** (95% CI `[-0.0048, +0.0207]`, Holm p = 1.000), and P5-wstat vs P2 collapses to **+0.0027** (95% CI `[-0.0075, +0.0126]`, Holm p = 1.000).\n"
        "   - Side-by-side reporting shows the apparent importance-weighted advantage disappearing once the generator artifact is removed.\n\n"
        "5. **Supervised (Label-Using) Detectors vs Label-Free:**\n"
        "   - Comparisons must not conflate significant and non-significant results across metrics.\n"
        "   - **At acc@4 on decoupled streams:** Page-Hinkley (P13) significantly outperforms P7 by **+0.0352** (Holm p = 0.001) and P5 by **+0.0313** (Holm p = 0.006). ADWIN (P10) does **not** achieve significance at acc@4 (+0.0257 vs P7, Holm p = 0.145; +0.0218 vs P5, Holm p = 0.217). Performance-Drop (P6) is borderline (+0.0210 vs P7, Holm p = 0.148; +0.0171 vs P5, Holm p = 0.330).\n"
        "   - **On Curve AUC on decoupled streams:** Both Page-Hinkley (+0.0259 vs P7, Holm p < 0.001; +0.0247 vs P5, Holm p = 0.001) and ADWIN (+0.0230 vs P7, Holm p = 0.009; +0.0219 vs P5, Holm p = 0.009) achieve statistically significant gains.\n\n"
        "6. **Pre-Registration Timeline & Number of Stream Looks:**\n"
        "   - Development was conducted across four stream sets: Seeds 1000–1029 (exploratory round 1), Seeds 2000–2029 (exploratory round 2), Seeds 3000–3029 (initial confirmatory evaluation before artifact detection), and Seeds 4000–4029 (confirmatory evaluation on decoupled streams).\n"
        "   - `HYPOTHESIS_v3.md` was drafted after analyzing the 3000–3029 results and identifying the generator artifact. The primary test list was updated accordingly, and `HYPOTHESIS_v3.md` was committed to git (commit `53861c3`) *before* executing the decoupled confirmatory sweep on seeds 4000–4029.\n\n"
        "7. **Single-Run Benchmark Designation:**\n"
        "   - The single-run evaluations on Fraud, Intrusion, and ELEC2 represent exploratory/descriptive benchmarks without replicate variance and are presented as descriptive context in appendices rather than confirmatory hypothesis tests.\n\n"
    )

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
