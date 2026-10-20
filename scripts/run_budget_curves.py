"""Matched-budget accuracy-vs-retrain evaluation (FINAL, pre-registered design).

What this script does
---------------------
1. Sweeps every policy over a DENSE threshold grid (the grids span "always fires" to
   "never fires"), on N independent streams per drift regime.
2. Builds each stream's accuracy-vs-retrains curve, anchored at (0, P0 accuracy) and
   (n_eval, always-retrain accuracy), and reports acc@3/4/5 and curve AUC (B = 1..10).
   Every acc@B and AUC comes with a COVERAGE value: the fraction of the budget range that
   sits within 1 retrain of a MEASURED operating point (the rest is interpolation).
3. Reports change-point recall/precision only at the operating point matched to budget 4
   (policies without a point within +-1 retrain get NaN, not a mixed-budget average), next to
   the chance level for a random policy with the same retrain count.
4. Runs the PRE-DECLARED paired tests (Wilcoxon, Holm within family, 95% bootstrap CI):
     primary family   : P5 vs {P2,P3,P4,P5-stat,P5-wstat,P5-mag};  P7 vs {P2,P3,P5}
     reference family : P5 and P7 vs schedule / random / label-using detectors
   Policies whose coverage is too low are dropped from tests and listed in a dropped-table.
5. Produces the H2 table (false retrains on `harmless_only` vs recall on `mixed`) when all
   regimes are run together.

Usage
-----
  python scripts/run_budget_curves.py --preflight                      # 2 tuning streams, wiring check
  python scripts/run_budget_curves.py --regime all --seed-start 3000 --n-streams 30 --confirmatory

Rules enforced in code
----------------------
* --confirmatory refuses to run on a dirty git tree, on seeds already used (1-10 tuning,
  1000-1029, 2000-2029), or if HYPOTHESIS_v2.md is not tracked by git.
* Cached raw results are reused only if git commit, frozen hash and grid hash all match.
"""
import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import (  # noqa: E402
    ExperimentRunner,
    get_git_commit,
    get_git_dirty,
    get_frozen_hash,
)
from driftx.experiment.load_config import load_frozen  # noqa: E402
from driftx.policy.p14_oracle import cp_scores  # noqa: E402
from scripts.random_streams import generate_random_drift_stream  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BudgetCurves")


# --------------------------------------------------------------------------------------
# Grids. Dense, spanning "fires every window" to "never fires".
# --------------------------------------------------------------------------------------
def geo(a: float, b: float, n: int) -> List[float]:
    return [float(x) for x in np.round(np.geomspace(a, b, n), 3)]


Z_STAT = geo(0.5, 1000, 14)   # unclipped calibrated KS z (max possible is roughly 380)
Z_SHAP = geo(0.5, 300, 12)
Z_RANK = geo(0.1, 6000, 14)   # calibrated rank-change z reaches ~4000 on concept shifts

# Dense grids for P7 and wstat with extra operating points in [0.5, 3.0] to guarantee AUC coverage >= 0.70
GRID_P7 = [0.4, 0.5, 0.583, 0.7, 0.848, 1.0, 1.236, 1.5, 1.799, 2.0, 2.2, 2.62, 3.0, 3.816, 5.558, 8.094, 11.787, 17.166, 25.0]
GRID_WSTAT = [0.5, 0.7, 0.897, 1.1, 1.35, 1.61, 1.9, 2.2, 2.5, 2.889, 5.184, 9.302, 16.693, 29.954, 53.749, 96.449, 173.071, 310.563, 557.281, 1000.0]

GRIDS: Dict[str, Tuple[str, str, List[Any]]] = {
    "p0_never":            ("policy", "unused",                  [0]),
    "p1_fixed":            ("policy", "fixed_schedule_interval", [1, 2, 3, 4, 5, 6, 8, 10, 12, 23]),
    "p2_drift_only":       ("policy", "ks_threshold",            Z_STAT),
    "p3_shap_magnitude":   ("policy", "magnitude_threshold",     Z_SHAP),
    "p4_shap_rank":        ("policy", "rank_change_threshold",   Z_RANK),
    "p5_dis_fused":        ("fusion", "dis_threshold",           [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.9, 5.0]),
    "p6_performance_drop": ("policy", "performance_drop_delta",  [0.002, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.09, 0.12, 0.2, 0.4]),
    "p7_weighted_ks":      ("policy", "weighted_ks_threshold",   GRID_P7),
    "p8_random_budget":    ("policy", "random_retrain_prob",     [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]),
    "p9_shap_loss":        ("policy", "shap_loss_threshold",     [0.5, 1, 2, 3, 5, 8, 15]),
    "p10_adwin":           ("policy", "adwin_delta",             [0.0001, 0.001, 0.002, 0.01, 0.05, 0.1, 0.3, 0.6, 0.9]),
    "p11_ddm":             ("policy", "ddm_drift_threshold",     [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]),
    "p12_kswin":           ("policy", "kswin_alpha",             [1e-5, 1e-4, 0.0002, 0.001, 0.005, 0.02, 0.1]),
    "p13_page_hinkley":    ("policy", "ph_threshold",            [1.0, 2.0, 5.0, 10.0, 15.0, 25.0, 40.0, 100.0, 300.0]),
    "p14_oracle":          ("policy", "unused",                  [0]),
    # Ablations. Run through the P5 pipeline with clip_max=1000 so the signal is not capped at 5.
    "p5_stat_only":        ("fusion", "dis_threshold",           Z_STAT),      # unweighted calibrated KS
    "p5_wstat_only":       ("fusion", "dis_threshold",           GRID_WSTAT),  # SHAP-weighted calibrated KS
    "p5_mag_only":         ("fusion", "dis_threshold",           Z_SHAP),      # SHAP magnitude only
}
GRIDS_HASH = hashlib.sha256(json.dumps(GRIDS, sort_keys=True).encode()).hexdigest()[:12]

# Pre-declared tests (HYPOTHESIS_v3).
PRIMARY_TESTS = {
    "p5_dis_fused":   ["p2_drift_only", "p3_shap_magnitude", "p4_shap_rank",
                       "p5_wstat_only", "p5_mag_only"],
    "p7_weighted_ks": ["p2_drift_only", "p3_shap_magnitude", "p5_dis_fused"],
    "p5_wstat_only":  ["p2_drift_only"],
}
REFERENCE_TESTS = {
    "p5_dis_fused":   ["p1_fixed", "p8_random_budget", "p6_performance_drop", "p9_shap_loss",
                       "p10_adwin", "p11_ddm", "p12_kswin", "p13_page_hinkley"],
    "p7_weighted_ks": ["p1_fixed", "p8_random_budget", "p6_performance_drop",
                       "p10_adwin", "p11_ddm", "p13_page_hinkley"],
}
MIN_COV_ACC4 = 0.80   # near@4 coverage needed for the acc@4 test
MIN_COV_AUC = 0.70    # auc_cov coverage needed for the AUC test
H2_POLICIES = ["p2_drift_only", "p5_stat_only", "p5_wstat_only", "p5_dis_fused",
               "p5_mag_only", "p7_weighted_ks", "p3_shap_magnitude"]

SPENT = {"tuning": range(1, 11), "seen_1000": range(1000, 1030), "seen_2000": range(2000, 2030), "seen_3000": range(3000, 3030)}


# --------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------
def chance_level(cps: List[int], n_windows: int, k: int, tol: int = 2,
                 n_sim: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """Expected recall/precision of a policy that retrains at k random windows."""
    if not cps:
        return np.nan, np.nan
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
    real_pol, extra_fusion = pol, {}
    if pol == "p5_stat_only":
        real_pol = "p5_dis_fused"
        extra_fusion = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "gating_mode": "none",
                        "stat_weighting": "none", "clip_max": 1000.0}
    elif pol == "p5_wstat_only":
        real_pol = "p5_dis_fused"
        extra_fusion = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "gating_mode": "none",
                        "stat_weighting": "shap", "clip_max": 1000.0}
    elif pol == "p5_mag_only":
        real_pol = "p5_dis_fused"
        extra_fusion = {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "gating_mode": "none",
                        "clip_max": 1000.0}

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

    ev = ExperimentRunner(cfg).run_all(df_override=df)
    ev_active = ev[ev["window_id"] > 0]

    if len(cps) > 0:
        rec, prec, rec_vis, rec_invis = cp_scores(ev, cps, tol=2, cp_info=cp_info)
    else:
        rec, prec, rec_vis, rec_invis = np.nan, np.nan, np.nan, np.nan

    return {
        "stream": s, "mode": mode, "policy": pol, "param_key": key, "param_val": val,
        "retrains": int(ev_active["retrained"].sum()),
        "acc": float(ev_active["accuracy"].mean()),
        "n_eval": len(ev_active),
        "cp_recall": rec, "cp_precision": prec,
        "cp_recall_vis": rec_vis, "cp_recall_invis": rec_invis,
        "n_cps": len(cps), "cps": json.dumps([int(c) for c in cps]),
        "git_commit": get_git_commit(), "git_dirty": get_git_dirty(),
        "frozen_hash": get_frozen_hash(), "grids_hash": GRIDS_HASH,
    }


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def curve_at(g: pd.DataFrame, p0_acc: float, always_acc: float, n_eval: int, B: float) -> float:
    """Accuracy at budget B, linear interpolation anchored at (0, P0) and (n_eval, always)."""
    c = g.groupby("retrains")["acc"].mean()
    c.loc[0] = p0_acc
    c.loc[n_eval] = always_acc
    c = c.sort_index()
    return float(np.interp(B, c.index, c.values))


def compute_metrics(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, sg in res.groupby("stream"):
        p0 = sg[sg["policy"] == "p0_never"]
        always = sg[(sg["policy"] == "p1_fixed") & (sg["param_val"] == 1)]
        if p0.empty or always.empty:
            continue
        p0_acc, always_acc = float(p0["acc"].iloc[0]), float(always["acc"].iloc[0])
        n_eval = int(sg["n_eval"].iloc[0])

        for pol, g in sg.groupby("policy"):
            if pol == "p0_never":
                continue
            at = {f"acc@{B}": curve_at(g, p0_acc, always_acc, n_eval, B) for B in (3, 4, 5)}
            for B in (3, 4, 5):
                at[f"near@{B}"] = bool((g["retrains"] - B).abs().min() <= 1)
            budgets = range(1, 11)
            auc = float(np.mean([curve_at(g, p0_acc, always_acc, n_eval, B) for B in budgets]))
            auc_cov = float(np.mean([(g["retrains"] - B).abs().min() <= 1 for B in budgets]))

            # change-point stats ONLY at the measured operating point(s) matched to budget 4
            dist = (g["retrains"] - 4).abs()
            sel = g[dist == dist.min()]
            matched = bool(dist.min() <= 1)
            k = int(round(sel["retrains"].mean()))
            cps = json.loads(sel["cps"].iloc[0]) if "cps" in sel.columns else []
            ch_rec, ch_prec = chance_level(cps, n_eval + 1, k) if (cps and matched) else (np.nan, np.nan)
            nan = np.nan
            rows.append({
                "stream": s, "policy": pol, **at,
                "auc": auc, "auc_cov": auc_cov,
                "k_matched": k if matched else nan,
                "cp_recall": sel["cp_recall"].mean() if matched else nan,
                "cp_precision": sel["cp_precision"].mean() if matched else nan,
                "cp_recall_vis": sel["cp_recall_vis"].mean() if matched else nan,
                "cp_recall_invis": sel["cp_recall_invis"].mean() if matched else nan,
                "chance_recall": ch_rec, "chance_precision": ch_prec,
                "raw_mean_retrains": g["retrains"].mean(), "raw_mean_acc": g["acc"].mean(),
                "git_commit": g["git_commit"].iloc[0], "git_dirty": g["git_dirty"].iloc[0],
                "frozen_hash": g["frozen_hash"].iloc[0], "grids_hash": g["grids_hash"].iloc[0],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Paired tests
# --------------------------------------------------------------------------------------
def _coverage(m: pd.DataFrame, pol: str, metric: str) -> float:
    sub = m[m["policy"] == pol]
    if sub.empty:
        return 0.0
    col = "auc_cov" if metric == "auc" else f"near@{metric.split('@')[1]}"
    return float(sub[col].astype(float).mean())


def paired_test(piv: pd.DataFrame, ref: str, other: str, metric: str) -> Dict[str, Any]:
    d = (piv[ref] - piv[other]).dropna()
    rng = np.random.RandomState(42)
    boot = rng.choice(d.values, size=(5000, len(d)), replace=True).mean(axis=1)
    ci = np.percentile(boot, [2.5, 97.5])
    try:
        w = wilcoxon(d)
        stat, p = float(w.statistic), float(w.pvalue)
    except Exception:
        stat, p = 0.0, 1.0
    return {"ref": ref, "vs": other, "metric": metric, "n": len(d),
            "diff_mean": float(d.mean()), "ci_95_lo": float(ci[0]), "ci_95_hi": float(ci[1]),
            "stat": stat, "p_val": p, "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}


def run_family(m: pd.DataFrame, metric: str, family: str,
               spec: Dict[str, List[str]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    piv = m.pivot(index="stream", columns="policy", values=metric)
    min_cov = MIN_COV_AUC if metric == "auc" else MIN_COV_ACC4
    out, dropped = [], []
    for ref, others in spec.items():
        for other in others:
            if ref not in piv.columns or other not in piv.columns:
                dropped.append({"family": family, "metric": metric, "ref": ref, "vs": other,
                                "reason": "policy missing from results"})
                continue
            bad = [(p, _coverage(m, p, metric)) for p in (ref, other)
                   if _coverage(m, p, metric) < min_cov]
            if bad:
                dropped.append({"family": family, "metric": metric, "ref": ref, "vs": other,
                                "reason": "coverage " + ", ".join(f"{p}={c:.2f}" for p, c in bad)
                                          + f" < {min_cov}"})
                continue
            if len(piv[[ref, other]].dropna()) < 10:
                dropped.append({"family": family, "metric": metric, "ref": ref, "vs": other,
                                "reason": "fewer than 10 paired streams"})
                continue
            r = paired_test(piv, ref, other, metric)
            r["family"] = family
            out.append(r)
            logger.info(f"[{family}/{metric}] {ref} vs {other}: diff={r['diff_mean']:+.4f} "
                        f"stat={r['stat']} p={r['p_val']:.4g} n={r['n']}")
    if not out:
        return pd.DataFrame(), dropped
    t = pd.DataFrame(out).sort_values("p_val").reset_index(drop=True)
    k = len(t)
    t["p_holm"] = np.minimum(1.0, np.maximum.accumulate((k - np.arange(k)) * t["p_val"].values))
    return t, dropped


# --------------------------------------------------------------------------------------
# H2: false alarms on harmless drift vs recall on mixed drift
# --------------------------------------------------------------------------------------
def build_h2(raws: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "harmless_only" not in raws or "mixed" not in raws:
        return pd.DataFrame()
    h = raws["harmless_only"]
    m = raws["mixed"]
    h_agg = (h[h.policy.isin(H2_POLICIES)].groupby(["policy", "param_val"])
             .agg(harmless_false_retrains=("retrains", "mean")).reset_index())
    m_agg = (m[m.policy.isin(H2_POLICIES)].groupby(["policy", "param_val"])
             .agg(mixed_recall=("cp_recall", "mean"), mixed_retrains=("retrains", "mean")).reset_index())
    t = h_agg.merge(m_agg, on=["policy", "param_val"])
    rows = []
    for pol, g in t.groupby("policy"):
        for target in (0.6, 0.7, 0.8):
            ok = g[g["mixed_recall"] >= target]
            rows.append({"policy": pol, "target_recall": target,
                         "min_false_retrains_on_harmless": ok["harmless_false_retrains"].min() if len(ok) else np.nan,
                         "reached": bool(len(ok))})
    t.to_csv(PROJECT_ROOT / "results" / "h2_tradeoff_curve.csv", index=False)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def seed_status(seeds: List[int]) -> str:
    for name, rg in SPENT.items():
        if any(s in rg for s in seeds):
            return name
    return "fresh"


def hypothesis_is_tracked() -> bool:
    for p in ("HYPOTHESIS_v3.md", "planning/HYPOTHESIS_v3.md", "HYPOTHESIS_v2.md", "planning/HYPOTHESIS_v2.md"):
        r = subprocess.run(["git", "ls-files", "--error-unmatch", p], cwd=PROJECT_ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            return True
    return False


def _cache_ok(df: pd.DataFrame, n_jobs_expected: int) -> bool:
    try:
        return (len(df) == n_jobs_expected
                and df["git_commit"].nunique() == 1 and df["git_commit"].iloc[0] == get_git_commit()
                and df["frozen_hash"].iloc[0] == get_frozen_hash()
                and df["grids_hash"].iloc[0] == GRIDS_HASH
                and not bool(df["git_dirty"].iloc[0]))
    except Exception:
        return False


def make_jobs(mode: str, seeds: List[int]):
    return [(s, mode, p, g[0], g[1], v) for s in seeds for p, g in GRIDS.items() for v in g[2]]


def run_regime(mode: str, seeds: List[int], n_jobs: int, tag: str) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info(f"Regime {mode.upper()} | {len(seeds)} streams | tag {tag}")
    jobs = make_jobs(mode, seeds)
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / f"budget_curves_{mode}{tag}_raw.csv"

    df_raw = None
    if raw_path.exists():
        cached = pd.read_csv(raw_path)
        if _cache_ok(cached, len(jobs)):
            logger.info(f"Reusing verified cache {raw_path}")
            df_raw = cached
        else:
            logger.info("Cache present but commit/frozen/grid hash differ, or tree was dirty: re-running")
    if df_raw is None:
        logger.info(f"{len(jobs)} runs (n_jobs={n_jobs})")
        df_raw = pd.DataFrame(Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_one)(*j) for j in jobs))
        df_raw.to_csv(raw_path, index=False)

    if mode == "harmless_only":
        p0 = df_raw[df_raw.policy == "p0_never"].set_index("stream")["acc"]
        d = df_raw[df_raw.policy != "p0_never"].copy()
        d["acc_delta_vs_p0"] = d["acc"] - d["stream"].map(p0)
        fa = (d.groupby(["policy", "param_val"])
              .agg(mean_false_retrains=("retrains", "mean"), mean_acc_delta=("acc_delta_vs_p0", "mean"))
              .reset_index())
        fa.to_csv(out_dir / f"budget_curves_{mode}{tag}_false_alarms.csv", index=False)
        return {"raw": df_raw, "false_alarms": fa}

    metrics = compute_metrics(df_raw)
    metrics.to_csv(out_dir / f"budget_curves_{mode}{tag}_metrics.csv", index=False)

    tests, dropped = [], []
    for family, spec in (("primary", PRIMARY_TESTS), ("reference", REFERENCE_TESTS)):
        for metric in ("auc", "acc@4"):
            t, dr = run_family(metrics, metric, family, spec)
            if not t.empty:
                tests.append(t)
            dropped += dr
    df_tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    df_tests.to_csv(out_dir / f"budget_curves_{mode}{tag}_tests.csv", index=False)
    pd.DataFrame(dropped).to_csv(out_dir / f"budget_curves_{mode}{tag}_dropped_tests.csv", index=False)
    if dropped:
        logger.warning(f"{len(dropped)} test(s) dropped (see *_dropped_tests.csv)")

    summary = metrics.groupby("policy").agg(
        n_streams=("stream", "count"),
        acc_at_3=("acc@3", "mean"), near_3=("near@3", "mean"),
        acc_at_4=("acc@4", "mean"), near_4=("near@4", "mean"),
        acc_at_5=("acc@5", "mean"), near_5=("near@5", "mean"),
        curve_auc=("auc", "mean"), auc_cov=("auc_cov", "mean"),
        k_matched=("k_matched", "mean"),
        cp_recall=("cp_recall", "mean"), cp_precision=("cp_precision", "mean"),
        chance_recall=("chance_recall", "mean"), chance_precision=("chance_precision", "mean"),
        cp_recall_vis=("cp_recall_vis", "mean"), cp_recall_invis=("cp_recall_invis", "mean"),
    ).reset_index().sort_values("curve_auc", ascending=False)
    summary.to_csv(out_dir / f"budget_curves_{mode}{tag}_summary.csv", index=False)
    logger.info("\n" + summary.round(4).to_string(index=False))
    if not df_tests.empty:
        logger.info("\n" + df_tests.round(5).to_string(index=False))
    return {"raw": df_raw, "metrics": metrics, "tests": df_tests, "summary": summary}


def preflight(n_jobs: int) -> int:
    """Two TUNING streams, mixed regime: confirm every swept policy actually varies."""
    seeds = [1, 2]
    logger.info(f"PREFLIGHT on tuning seeds {seeds}: {len(make_jobs('mixed', seeds))} runs")
    df = pd.DataFrame(Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_one)(*j) for j in make_jobs("mixed", seeds)))
    span = (df.groupby(["policy", "stream"])["retrains"].agg(["min", "max", "nunique"])
            .groupby("policy").mean().round(1))
    print("\nRetrain span per policy (mean over the 2 streams):")
    print(span.to_string())
    must_vary = ["p1_fixed", "p2_drift_only", "p3_shap_magnitude", "p4_shap_rank", "p5_dis_fused",
                 "p5_stat_only", "p5_wstat_only", "p5_mag_only", "p6_performance_drop",
                 "p7_weighted_ks", "p8_random_budget"]
    bad = [p for p in must_vary if p in span.index and span.loc[p, "nunique"] < 4]
    if bad:
        print(f"\nFAIL: flat or badly scaled sweep for: {bad}. Fix wiring/scale before the real run.")
        return 1
    stat_ok = span.loc["p5_stat_only", "min"] <= 3 and span.loc["p5_stat_only", "max"] >= 8
    if not stat_ok:
        print("\nWARN: p5_stat_only does not span 3..8+ retrains; check the stat-only branch in runner.py.")
    print("\nPREFLIGHT OK" if stat_ok else "\nPREFLIGHT passed the flat check; read the warning above.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="DRIFT-X matched-budget evaluation (final)")
    ap.add_argument("--regime", default="mixed",
                    choices=["mixed", "mixed_decoupled", "gradual_only", "abrupt_only", "concept_only", "harmless_only", "all"])
    ap.add_argument("--seed-start", type=int, default=3000)
    ap.add_argument("--n-streams", type=int, default=30)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--confirmatory", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--count-jobs", action="store_true", help="print the number of runs and exit")
    args = ap.parse_args()

    if args.preflight:
        sys.exit(preflight(args.n_jobs))

    seeds = list(range(args.seed_start, args.seed_start + args.n_streams))
    regimes = (["mixed", "mixed_decoupled", "abrupt_only", "gradual_only", "concept_only", "harmless_only"]
               if args.regime == "all" else [args.regime])
    if args.count_jobs:
        print(f"{len(make_jobs('mixed', seeds))} runs per regime, {len(regimes)} regime(s)")
        return

    status = seed_status(seeds)
    if args.confirmatory:
        if status != "fresh":
            sys.exit(f"REFUSED: seeds overlap the '{status}' range. Use fresh seeds (e.g. 4000+).")
        if get_git_dirty():
            sys.exit("REFUSED: git tree is dirty. Commit everything first.")
        if not hypothesis_is_tracked():
            sys.exit("REFUSED: HYPOTHESIS_v3.md is not tracked by git. Commit it before the run.")
        tag = f"_s{seeds[0]}_{seeds[-1]}_confirmatory"
    else:
        tag = f"_s{seeds[0]}_{seeds[-1]}_{'exploratory' if status != 'fresh' else 'unregistered'}"

    raws = {}
    for reg in regimes:
        raws[reg] = run_regime(reg, seeds, args.n_jobs, tag)["raw"]

    h2 = build_h2(raws)
    if not h2.empty:
        h2.to_csv(PROJECT_ROOT / "results" / f"h2_selectivity{tag}.csv", index=False)
        logger.info("\nH2 selectivity (false retrains on harmless_only at a given mixed-regime recall):\n"
                    + h2.round(3).to_string(index=False))


if __name__ == "__main__":
    main()