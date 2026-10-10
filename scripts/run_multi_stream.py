"""Multi-stream evaluation across 30 randomized non-stationary streams.

Uses frozen configuration (configs/frozen.yaml) with seeds 1000-1029.
Computes paired Wilcoxon signed-rank tests and 5,000 bootstrap CIs.
Saves raw results and formatted tables for publication.
"""
import copy
import hashlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftx.experiment.runner import ExperimentRunner
from scripts.random_streams import generate_random_drift_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiStreamEval")


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit[:12]
    except Exception:
        return "unknown"


def get_config_hash(cfg_path: Path) -> str:
    try:
        hasher = hashlib.sha256()
        with open(cfg_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()[:12]
    except Exception:
        return "unknown"


def run_multi_stream_eval(
    n_streams: int = 30,
    start_seed: int = 1000,
    config_path: Path = PROJECT_ROOT / "configs" / "frozen.yaml",
):
    if not config_path.exists():
        config_path = PROJECT_ROOT / "configs" / "default.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    git_sha = get_git_commit()
    cfg_hash = get_config_hash(config_path)
    logger.info(f"Loaded config from {config_path} (hash={cfg_hash}, git={git_sha})")

    active_policies = base_config.get("policy", {}).get("active_policies", [
        "p0_never", "p1_fixed", "p2_drift_only", "p3_shap_magnitude",
        "p4_shap_rank", "p5_dis_fused", "p6_performance_drop",
        "p7_weighted_ks", "p8_random_budget", "p9_shap_loss"
    ])

    rows = []
    stream_seeds = list(range(start_seed, start_seed + n_streams))

    logger.info(f"Evaluating {len(active_policies)} policies across {n_streams} streams (seeds {stream_seeds[0]}..{stream_seeds[-1]})...")

    for s_idx, s in enumerate(stream_seeds, 1):
        df, cps = generate_random_drift_stream(seed=s, n_windows=24, n=1500, p=15, drift_mode="mixed")
        cfg = copy.deepcopy(base_config)
        cfg["project"]["seed"] = s
        cfg["project"]["n_seeds"] = 1
        cfg["policy"]["active_policies"] = active_policies
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
                "config_hash": cfg_hash,
                "git_commit": git_sha,
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
            stat, p_val = wilcoxon(diff_acc)
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
    report.append(f"**Provenance:** Git Commit: `{git_sha}` | Config Hash: `{cfg_hash}`\n")
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

    print("\n" + "".join(report))
    return df_all, df_stats


if __name__ == "__main__":
    run_multi_stream_eval(n_streams=30, start_seed=1000)
