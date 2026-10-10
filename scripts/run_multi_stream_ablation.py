"""Ablation evaluation across the exact same 30 evaluation streams (seeds 1000-1029).

Variants tested on each stream:
1. full_dis  (alpha=0.4, beta=0.4, gamma=0.2, gating_mode=two_channel)
2. stat_only (alpha=1.0, beta=0.0, gamma=0.0, gating_mode=none)
3. mag_only  (alpha=0.0, beta=1.0, gamma=0.0, gating_mode=none)
4. rank_only (alpha=0.0, beta=0.0, gamma=1.0, gating_mode=none)
5. stat_mag  (alpha=0.5, beta=0.5, gamma=0.0, gating_mode=two_channel)
"""
import copy
import logging
import sys
from pathlib import Path
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
logger = logging.getLogger("MultiStreamAblation")

ABLATION_VARIANTS = {
    "full_dis":  {"alpha": 0.4, "beta": 0.4, "gamma": 0.2, "gating_mode": "two_channel"},
    "stat_only": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "gating_mode": "none"},
    "mag_only":  {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "gating_mode": "none"},
    "rank_only": {"alpha": 0.0, "beta": 0.0, "gamma": 1.0, "gating_mode": "none"},
    "stat_mag":  {"alpha": 0.5, "beta": 0.5, "gamma": 0.0, "gating_mode": "two_channel"},
}


def run_multi_stream_ablation(n_streams: int = 30, start_seed: int = 1000):
    cfg_path = PROJECT_ROOT / "configs" / "frozen.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    rows = []
    stream_seeds = list(range(start_seed, start_seed + n_streams))
    logger.info(f"Running multi-stream ablation across {n_streams} streams (seeds {stream_seeds[0]}..{stream_seeds[-1]})...")

    for s_idx, s in enumerate(stream_seeds, 1):
        df, cps = generate_random_drift_stream(seed=s, n_windows=24, n=1500, p=15, drift_mode="mixed")
        for variant_name, weights in ABLATION_VARIANTS.items():
            cfg = copy.deepcopy(base_cfg)
            cfg["project"]["seed"] = s
            cfg["project"]["n_seeds"] = 1
            cfg["policy"]["active_policies"] = ["p5_dis_fused"]
            cfg["fusion"].update(weights)

            runner = ExperimentRunner(cfg)
            res = runner.run_all(df_override=df, output_filename=f"abl_temp_{s}.csv")
            eval_rows = res[res["window_id"] > 0]
            acc = eval_rows["accuracy"].mean()
            retrains = eval_rows["retrained"].sum()

            rows.append({
                "stream": s,
                "variant": variant_name,
                "accuracy": float(acc),
                "net_retrains": int(retrains),
                "n_changepoints": len(cps),
            })
        logger.info(f"Ablation stream {s_idx}/{n_streams} (seed {s}) complete.")

    df_all = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / "multi_stream_ablation_raw.csv"
    df_all.to_csv(raw_path, index=False)

    summary = df_all.groupby("variant").agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_retrains=("net_retrains", "mean"),
        std_retrains=("net_retrains", "std"),
    ).round(4).reset_index()
    summary_path = out_dir / "multi_stream_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Paired Wilcoxon and bootstrap vs full_dis
    piv_acc = df_all.pivot(index="stream", columns="variant", values="accuracy")
    piv_ret = df_all.pivot(index="stream", columns="variant", values="net_retrains")

    stat_records = []
    for other in [v for v in ABLATION_VARIANTS.keys() if v != "full_dis"]:
        diff_acc = piv_acc["full_dis"] - piv_acc[other]
        diff_ret = piv_ret["full_dis"] - piv_ret[other]
        try:
            stat, p_val = wilcoxon(diff_acc)
        except Exception:
            stat, p_val = float("nan"), float("nan")

        rng = np.random.RandomState(42)
        diff_vals = diff_acc.values
        boot_means = [rng.choice(diff_vals, size=len(diff_vals), replace=True).mean() for _ in range(5000)]
        ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

        stat_records.append({
            "Comparator": other,
            "Mean Acc Diff (Full DIS - Other)": float(diff_acc.mean()),
            "95% Bootstrap CI": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            "Wilcoxon p-value": float(p_val),
            "Mean Retrains Diff": float(diff_ret.mean()),
            "Full DIS Wins Count": int((diff_acc > 0).sum()),
            "Total Streams": len(diff_acc),
        })

    df_stats = pd.DataFrame(stat_records)
    stat_path = out_dir / "multi_stream_ablation_wilcoxon.csv"
    df_stats.to_csv(stat_path, index=False)

    print("\n=== Multi-Stream Ablation Summary Across 30 Streams ===")
    print(summary.to_string(index=False))
    print("\n=== Paired Statistical Tests vs Full DIS ===")
    print(df_stats.to_string(index=False))


if __name__ == "__main__":
    run_multi_stream_ablation()
