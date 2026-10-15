"""Matched-Budget Accuracy-vs-Retrain Curves & Hypothesis Testing.

Constructs empirical accuracy-vs-retrains curves across drift streams anchored at both ends:
- Left anchor: (0, P0 accuracy)
- Right anchor: (N_eval, Always-Retrain accuracy from P1 interval=1)

Evaluates matched-budget accuracy (acc@3, acc@4, acc@5), curve AUC, change-point recall
decomposed into visible (covariate-assisted) vs invisible (pure concept), change-point precision,
and paired Wilcoxon tests with Holm-Bonferroni correction and 95% bootstrap CIs.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import (
    ExperimentRunner,
    get_git_commit,
    get_git_dirty,
    get_frozen_hash,
    compute_config_hash,
)
from driftx.experiment.load_config import load_frozen
from driftx.policy.p14_oracle import cp_scores
from scripts.random_streams import generate_random_drift_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BudgetCurves")

GRIDS = {
    "p0_never":            ("policy", "unused",                  [0]),
    "p1_fixed":            ("policy", "fixed_schedule_interval", [1, 2, 3, 4, 6, 8, 12, 23]),
    "p2_drift_only":       ("policy", "ks_threshold",            [1.0, 2.0, 5.0, 15.0, 50.0, 200.0]),
    "p3_shap_magnitude":   ("policy", "magnitude_threshold",     [1, 2, 3, 5, 8, 15, 30]),
    "p4_shap_rank":        ("policy", "rank_change_threshold",   [0.3, 0.6, 1, 2, 3, 5, 8, 15, 30]),
    "p5_dis_fused":        ("fusion", "dis_threshold",           [0.2, 0.5, 1.0, 2.0, 3.5, 4.0, 4.9, 5.0]),
    "p6_performance_drop": ("policy", "performance_drop_delta",  [0.005, 0.015, 0.03, 0.06, 0.15]),
    "p7_weighted_ks":      ("policy", "weighted_ks_threshold",   [0.6, 0.8, 1.0, 1.3, 1.8, 2.5, 4, 8, 14]),
    "p8_random_budget":    ("policy", "random_retrain_prob",     [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]),
    "p9_shap_loss":        ("policy", "shap_loss_threshold",     [1, 2, 3, 5, 8]),
    "p10_adwin":           ("policy", "adwin_delta",             [0.0002, 0.002, 0.02, 0.1, 0.3, 0.6, 0.9]),
    "p11_ddm":             ("policy", "ddm_drift_threshold",     [1.5, 2.5, 3.5, 5.0]),
    "p12_kswin":           ("policy", "kswin_alpha",             [0.0002, 0.001, 0.005, 0.02]),
    "p13_page_hinkley":    ("policy", "ph_threshold",            [5.0, 15.0, 40.0, 100.0]),
    "p14_oracle":          ("policy", "unused",                  [0]),
    "p5_stat_only":        ("fusion", "dis_threshold",           [0.2, 0.5, 1.0, 2.0, 3.5, 4.0, 4.9, 5.0]),
    "p5_wstat_only":       ("fusion", "dis_threshold",           [0.2, 0.5, 1.0, 2.0, 3.5, 4.0, 4.9, 5.0]),
    "p5_mag_only":         ("fusion", "dis_threshold",           [0.2, 0.5, 1.0, 2.0, 3.5, 4.0, 4.9, 5.0]),
}


def chance_level(cps: List[int], n_windows: int, k: int, tol: int = 2, n_sim: int = 1000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    cand = np.arange(1, n_windows)
    rec, prec = [], []
    for _ in range(n_sim):
        r = set(rng.choice(cand, size=min(k, len(cand)), replace=False)) if k > 0 else set()
        hit = {c for c in cps if any(c <= w <= c + tol for w in r)}
        good = {w for w in r if any(c <= w <= c + tol for c in cps)}
        rec.append(len(hit) / len(cps))
        if r:
            prec.append(len(good) / len(r))
    return float(np.mean(rec)), (float(np.mean(prec)) if prec else np.nan)


def run_one(s: int, mode: str, pol: str, sec: str, key: str, val: Any) -> Dict[str, Any]:
    df, cps, cp_info = generate_random_drift_stream(s, drift_mode=mode, return_info=True)
    real_pol = pol
    extra_fusion = {}
    if pol == "p5_stat_only":
        real_pol = "p5_dis_fused"
        extra_fusion = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "gating_mode": "none", "stat_weighting": "none"}
    elif pol == "p5_wstat_only":
        real_pol = "p5_dis_fused"
        extra_fusion = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "gating_mode": "none", "stat_weighting": "shap"}
    elif pol == "p5_mag_only":
        real_pol = "p5_dis_fused"
        extra_fusion = {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "gating_mode": "none"}

    cfg = load_frozen({
        "project": {"seed": s, "n_seeds": 1},
        "policy": {"active_policies": [real_pol]},
        "mlflow": {"enabled": False},
        "output": {"save_csv": False, "export_artifacts": False},
        "model": {"params": {"n_jobs": 1}},
    })
    if extra_fusion:
        cfg["fusion"].update(extra_fusion)
    if real_pol == "p14_oracle":
        cfg["policy"]["oracle_cps"] = [int(c) for c in cps]
    elif key != "unused":
        cfg[sec][key] = val
        if sec == "fusion" and key == "dis_threshold":
            cfg["fusion"]["fixed_threshold"] = val

    runner = ExperimentRunner(cfg)
    ev = runner.run_all(df_override=df)
    ev_active = ev[ev["window_id"] > 0]
    retrains = int(ev_active["retrained"].sum())
    acc = float(ev_active["accuracy"].mean())
    n_eval = len(ev_active)

    # Compute change point scores decomposed by visibility
    if len(cps) > 0:
        rec, prec, rec_vis, rec_invis = cp_scores(ev, cps, tol=2, cp_info=cp_info)
    else:
        rec, prec, rec_vis, rec_invis = np.nan, np.nan, np.nan, np.nan

    return {
        "stream": s,
        "mode": mode,
        "policy": pol,
        "param_key": key,
        "param_val": val,
        "retrains": retrains,
        "acc": acc,
        "n_eval": n_eval,
        "cp_recall": rec,
        "cp_precision": prec,
        "cp_recall_vis": rec_vis,
        "cp_recall_invis": rec_invis,
        "n_cps": len(cps),
        "cps": json.dumps([int(c) for c in cps]),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "frozen_hash": get_frozen_hash(),
    }


def curve_at(g: pd.DataFrame, p0_acc: float, always_acc: float, n_eval: int, B: float) -> float:
    """Interpolate accuracy at budget B anchored at (0, p0_acc) and (n_eval, always_acc)."""
    c = g.groupby("retrains")["acc"].mean()
    c.loc[0] = p0_acc
    c.loc[n_eval] = always_acc
    c = c.sort_index()
    return float(np.interp(B, c.index, c.values))


def compute_metrics(res: pd.DataFrame) -> pd.DataFrame:
    """Compute matched budget metrics (acc@3, acc@4, acc@5) and curve AUC using dual anchors."""
    rows = []
    # Identify per-stream anchors
    for s, s_group in res.groupby("stream"):
        p0_sub = s_group[s_group["policy"] == "p0_never"]
        always_sub = s_group[(s_group["policy"] == "p1_fixed") & (s_group["param_val"] == 1)]
        if p0_sub.empty or always_sub.empty:
            continue
        p0_acc = p0_sub["acc"].iloc[0]
        always_acc = always_sub["acc"].iloc[0]
        n_eval = s_group["n_eval"].iloc[0]

        for pol, g in s_group.groupby("policy"):
            if pol == "p0_never":
                continue
            at_budgets = {f"acc@{B}": curve_at(g, p0_acc, always_acc, n_eval, B) for B in (3, 4, 5)}
            for B in (3, 4, 5):
                at_budgets[f"near@{B}"] = bool((g["retrains"] - B).abs().min() <= 1)
            grid = [curve_at(g, p0_acc, always_acc, n_eval, B) for B in range(1, 11)]
            auc = float(np.mean(grid))

            B = 4
            row = g.loc[(g["retrains"] - B).abs().idxmin()]     # operating point closest to budget 4
            cps_val = row.get("cps", "[]")
            cps = json.loads(cps_val) if isinstance(cps_val, str) else [int(c) for c in (cps_val or [])]
            ch_rec, ch_prec = chance_level(cps, int(row["n_eval"]) + 1, int(row["retrains"])) if cps else (np.nan, np.nan)

            rows.append({
                "stream": s,
                "policy": pol,
                **at_budgets,
                "auc": auc,
                "k_matched": int(row["retrains"]),
                "cp_recall": row["cp_recall"],
                "cp_precision": row["cp_precision"],
                "cp_recall_vis": row["cp_recall_vis"],
                "cp_recall_invis": row["cp_recall_invis"],
                "chance_recall": ch_rec,
                "chance_precision": ch_prec,
                "raw_mean_retrains": g["retrains"].mean(),
                "raw_mean_acc": g["acc"].mean(),
                "git_commit": g["git_commit"].iloc[0] if "git_commit" in g.columns else get_git_commit(),
                "git_dirty": g["git_dirty"].iloc[0] if "git_dirty" in g.columns else get_git_dirty(),
                "frozen_hash": g["frozen_hash"].iloc[0] if "frozen_hash" in g.columns else get_frozen_hash(),
            })
    return pd.DataFrame(rows)


def run_paired_tests(m: pd.DataFrame, metric: str = "auc", ref: str = "p5_dis_fused") -> pd.DataFrame:
    """Run paired Wilcoxon tests vs ref with Holm correction and bootstrap CIs."""
    piv = m.pivot(index="stream", columns="policy", values=metric)
    if ref not in piv.columns:
        return pd.DataFrame()
    out = []
    # Drop p14_oracle from comparative tests (used as ceiling marker only)
    candidates = [c for c in piv.columns if c != ref and c != "p14_oracle"]
    near_col = f"near@{metric.split('@')[1]}" if "@" in metric else "near@4"
    for other in candidates:
        # Drop a policy from tests when its coverage is below 80%
        if near_col in m.columns:
            cov = m[m["policy"] == other][near_col].mean()
            if cov < 0.80:
                logger.warning(f"Dropping {other} from {metric} tests due to low coverage ({cov:.1%} < 80%)")
                continue

        d = (piv[ref] - piv[other]).dropna()
        if len(d) < 5:
            continue
        rng = np.random.RandomState(42)
        ci = np.percentile([rng.choice(d, len(d), replace=True).mean() for _ in range(5000)], [2.5, 97.5])
        stat = 0.0
        try:
            res_w = wilcoxon(d)
            stat, p_val = float(res_w.statistic), float(res_w.p_value)
        except Exception:
            p_val = 1.0

        logger.info(f"Paired Wilcoxon [{metric}] vs {other}: stat={stat}, p_val={p_val}, n={len(d)}")
        logger.debug(f"Differences head for {other} ({metric}):\n{d.head()}")

        out.append({
            "vs": other,
            "metric": metric,
            "n": len(d),
            "diff_mean": float(d.mean()),
            "ci_95_lo": float(ci[0]),
            "ci_95_hi": float(ci[1]),
            "stat": stat,
            "p_val": float(p_val),
            "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum()),
            "ties": int((d == 0).sum()),
        })
    if not out:
        return pd.DataFrame()
    t = pd.DataFrame(out).sort_values("p_val").reset_index(drop=True)
    m_ = len(t)
    t["p_holm"] = np.minimum(1.0, np.maximum.accumulate((m_ - np.arange(m_)) * t["p_val"]))
    return t


def run_regime(
    mode: str = "mixed",
    seeds: range = range(1000, 1030),
    n_jobs: int = 4,
    confirmatory: bool = False,
) -> Dict[str, Any]:
    logger.info(f"\n=======================================================")
    logger.info(f"Running Budget Curves for Regime: {mode.upper()} ({len(seeds)} streams)")
    logger.info(f"=======================================================")

    jobs = [
        (s, mode, p, g[0], g[1], v)
        for s in seeds
        for p, g in GRIDS.items()
        for v in g[2]
    ]
    logger.info(f"Total jobs to execute: {len(jobs)} (n_jobs={n_jobs})")
    results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_one)(*j) for j in jobs)
    df_raw = pd.DataFrame(results)

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    seed_list = list(seeds)
    s_start, s_end = seed_list[0], seed_list[-1]
    is_exploratory = any(s in range(1000, 1030) for s in seed_list) and not confirmatory
    tag = f"_s{s_start}_{s_end}" + ("_exploratory" if is_exploratory else "")

    raw_path = out_dir / f"budget_curves_{mode}_raw.csv"
    tagged_raw_path = out_dir / f"budget_curves_{mode}{tag}_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    df_raw.to_csv(tagged_raw_path, index=False)
    logger.info(f"Raw sweep results saved to {raw_path} and {tagged_raw_path}")

    # Harmless drift regime analysis
    if mode == "harmless_only":
        p0 = df_raw[df_raw.policy == "p0_never"].set_index("stream")["acc"]
        d = df_raw[df_raw.policy != "p0_never"].copy()
        d["acc_loss_vs_p0"] = d["acc"] - d["stream"].map(p0)
        fa = (d.groupby(["policy", "param_val"])
                .agg(mean_retrains=("retrains", "mean"), mean_acc_delta=("acc_loss_vs_p0", "mean"))
                .reset_index())
        fa_path = out_dir / f"budget_curves_{mode}_false_alarms.csv"
        tagged_fa_path = out_dir / f"budget_curves_{mode}{tag}_false_alarms.csv"
        fa.to_csv(fa_path, index=False)
        fa.to_csv(tagged_fa_path, index=False)
        logger.info(f"False alarms table saved to {fa_path} and {tagged_fa_path}")
        return {"raw": df_raw, "false_alarms": fa}

    # Guard: Fail loudly if any swept policy has a flat sweep
    min_nunique = 3.0 if len(seeds) >= 10 else 2.0
    for pol, g in df_raw.groupby("policy"):
        if g["param_key"].iloc[0] != "unused" and g.groupby("stream")["retrains"].nunique().mean() < min_nunique:
            raise RuntimeError(f"Flat sweep for {pol}: swept key is not wired into the runner or scale is wrong.")

    # Compute metrics (acc@3, acc@4, acc@5, AUC)
    df_metrics = compute_metrics(df_raw)
    metrics_path = out_dir / f"budget_curves_{mode}_metrics.csv"
    tagged_metrics_path = out_dir / f"budget_curves_{mode}{tag}_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    df_metrics.to_csv(tagged_metrics_path, index=False)
    logger.info(f"Metrics table saved to {metrics_path} and {tagged_metrics_path}")

    # Run paired statistical tests for AUC and acc@4
    tests_auc = run_paired_tests(df_metrics, metric="auc", ref="p5_dis_fused")
    tests_acc4 = run_paired_tests(df_metrics, metric="acc@4", ref="p5_dis_fused")
    df_tests = pd.concat([tests_auc, tests_acc4], ignore_index=True)
    tests_path = out_dir / f"budget_curves_{mode}_tests.csv"
    tagged_tests_path = out_dir / f"budget_curves_{mode}{tag}_tests.csv"
    df_tests.to_csv(tests_path, index=False)
    df_tests.to_csv(tagged_tests_path, index=False)
    logger.info(f"Paired test results saved to {tests_path} and {tagged_tests_path}")

    # Print summary table with coverage beside every acc@B
    summary = df_metrics.groupby("policy").agg(
        n_streams=("stream", "count"),
        acc_at_3=("acc@3", "mean"),
        near_3=("near@3", "mean"),
        acc_at_4=("acc@4", "mean"),
        near_4=("near@4", "mean"),
        acc_at_5=("acc@5", "mean"),
        near_5=("near@5", "mean"),
        curve_auc=("auc", "mean"),
        k_matched=("k_matched", "mean"),
        cp_recall=("cp_recall", "mean"),
        cp_precision=("cp_precision", "mean"),
        cp_recall_vis=("cp_recall_vis", "mean"),
        cp_recall_invis=("cp_recall_invis", "mean"),
        chance_recall=("chance_recall", "mean"),
        chance_precision=("chance_precision", "mean"),
    ).reset_index().sort_values("curve_auc", ascending=False)
    logger.info(f"\n--- Regime {mode.upper()} Summary ---")
    logger.info(summary.to_string(index=False))

    if not df_tests.empty:
        logger.info(f"\n--- Paired Tests (P5 vs Competitors) ---")
        logger.info(df_tests.to_string(index=False))

    return {
        "raw": df_raw,
        "metrics": df_metrics,
        "tests": df_tests,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="DRIFT-X Budget-Matched Curve Evaluation")
    parser.add_argument("--regime", type=str, default="mixed",
                        choices=["mixed", "gradual_only", "abrupt_only", "concept_only", "harmless_only", "all"],
                        help="Drift regime to evaluate")
    parser.add_argument("--seed-start", type=int, default=1000, help="Starting stream seed")
    parser.add_argument("--n-streams", type=int, default=30, help="Number of streams (default: 30)")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel worker processes (default: 4)")
    parser.add_argument("--confirmatory", action="store_true", help="Mark run as confirmatory (suppresses _exploratory suffix)")
    args = parser.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.n_streams)
    regimes = ["mixed", "gradual_only", "abrupt_only", "concept_only", "harmless_only"] if args.regime == "all" else [args.regime]

    for reg in regimes:
        run_regime(mode=reg, seeds=seeds, n_jobs=args.n_jobs, confirmatory=args.confirmatory)


if __name__ == "__main__":
    main()
