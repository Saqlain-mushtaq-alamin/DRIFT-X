"""Multi-stream evaluation across 30 randomized non-stationary streams.

Uses frozen configuration (configs/frozen.yaml) with seeds 1000-1029.
Computes paired Wilcoxon signed-rank tests and 5,000 bootstrap CIs.
Saves raw results and formatted tables for publication.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import (
    ExperimentRunner,
    get_git_commit,
    get_git_dirty,
    get_frozen_hash,
)
from driftx.experiment.load_config import load_frozen
from scripts.random_streams import generate_random_drift_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiStreamEval")


def run_multi_stream_eval(
    n_streams: int = 30,
    start_seed: int = 1000,
):
    base_config = load_frozen()
    git_sha = get_git_commit()
    git_dirty = get_git_dirty()
    f_hash = get_frozen_hash()
    logger.info(f"Loaded frozen config (frozen_hash={f_hash}, git={git_sha}, dirty={git_dirty})")

    active_policies = base_config.get("policy", {}).get("active_policies", [
        "p0_never", "p1_fixed", "p2_drift_only", "p3_shap_magnitude",
        "p4_shap_rank", "p5_dis_fused", "p6_performance_drop",
        "p7_weighted_ks", "p8_random_budget", "p9_shap_loss",
        "p10_adwin", "p11_ddm", "p12_kswin", "p13_page_hinkley"
    ])

    rows = []
    stream_seeds = list(range(start_seed, start_seed + n_streams))

    logger.info(f"Evaluating {len(active_policies)} policies across {n_streams} streams (seeds {stream_seeds[0]}..{stream_seeds[-1]})...")

    for s_idx, s in enumerate(stream_seeds, 1):
        df, cps = generate_random_drift_stream(seed=s, n_windows=24, n=1500, p=15, drift_mode="mixed")
        cfg = load_frozen({
            "project": {"seed": s, "n_seeds": 1},
            "policy": {"active_policies": active_policies},
            "mlflow": {"enabled": False},
            "output": {"save_csv": False, "export_artifacts": False},
            "model": {"params": {"n_jobs": 1}},
        })
        runner = ExperimentRunner(cfg)
        res = runner.run_all(df_override=df, output_filename=f"ms_temp_{s}.csv")
        eval_rows = res[res["window_id"] > 0]

        for pol in active_policies:
            pol_df = eval_rows[eval_rows["policy"] == pol]
            if pol_df.empty:
                continue
            mean_acc = pol_df["accuracy"].mean()
            net_retrains = pol_df["retrained"].sum()
            rows.append({
                "stream": s,
                "policy": pol,
                "accuracy": float(mean_acc),
                "net_retrains": int(net_retrains),
                "n_changepoints": len(cps),
                "frozen_hash": f_hash,
                "git_commit": git_sha,
                "git_dirty": git_dirty,
            })
        logger.info(f"Stream {s_idx}/{n_streams} (seed {s}) complete with {len(cps)} CPs.")

    df_all = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / "multi_stream_raw.csv"
    df_all.to_csv(raw_path, index=False)
    logger.info(f"Saved raw multi-stream results to {raw_path}")

    # Summary table per policy
    summary = df_all.groupby("policy").agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_retrains=("net_retrains", "mean"),
        std_retrains=("net_retrains", "std"),
    ).round(4).reset_index()
    summary_path = out_dir / "multi_stream_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")

    # Statistical comparison vs P5
    piv_acc = df_all.pivot(index="stream", columns="policy", values="accuracy")
    piv_ret = df_all.pivot(index="stream", columns="policy", values="net_retrains")

    stat_records = []
    comparators = [p for p in active_policies if p != "p5_dis_fused"]
    for other in comparators:
        if other not in piv_acc.columns or "p5_dis_fused" not in piv_acc.columns:
            continue
        diff_acc = piv_acc["p5_dis_fused"] - piv_acc[other]
        diff_ret = piv_ret["p5_dis_fused"] - piv_ret[other]
        
        # Paired Wilcoxon
        try:
            res_w = wilcoxon(diff_acc)
            stat, p_val = float(res_w.statistic), float(res_w.pvalue)
        except Exception:
            stat, p_val = float("nan"), float("nan")

        # 5,000 bootstrap CI for mean difference
        rng = np.random.RandomState(42)
        diff_vals = diff_acc.values
        boot_means = [rng.choice(diff_vals, size=len(diff_vals), replace=True).mean() for _ in range(5000)]
        ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

        stat_records.append({
            "Comparator": other,
            "Mean Acc Diff (P5 - Other)": float(diff_acc.mean()),
            "95% Bootstrap CI": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            "Wilcoxon stat": float(stat),
            "Wilcoxon p-value": float(p_val),
            "Mean Retrains Diff": float(diff_ret.mean()),
            "P5 Wins Count": int((diff_acc > 0).sum()),
            "Total Streams": len(diff_acc),
        })

    df_stats = pd.DataFrame(stat_records)
    stat_path = out_dir / "multi_stream_statistical_tests.csv"
    df_stats.to_csv(stat_path, index=False)

    # Format Markdown Report
    report = []
    report.append("# DRIFT-X 30-Stream Rigorous Evaluation Report\n")
    report.append(f"**Provenance:** Git Commit: `{git_sha}` | Dirty: `{git_dirty}` | Frozen Hash: `{f_hash}`\n")
    report.append(f"**Streams:** {n_streams} independent streams (seeds {start_seed}..{start_seed + n_streams - 1})\n")
    report.append(f"**Drift Regime:** Mixed harmless feature shifts + abrupt concept shifts\n\n")

    report.append("## 1. Policy Aggregate Performance Across 30 Streams\n")
    report.append(summary.to_markdown(index=False))
    report.append("\n\n")

    report.append("## 2. Paired Statistical Tests vs. DIS-Fused (P5)\n")
    report.append(df_stats.to_markdown(index=False))
    report.append("\n\n")

    report_path = out_dir / "multi_stream_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(report))
    logger.info(f"Saved full statistical report to {report_path}")

    return df_all, df_stats


if __name__ == "__main__":
    run_multi_stream_eval(n_streams=30, start_seed=1000)
