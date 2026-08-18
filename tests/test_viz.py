"""Unit tests for the DRIFT-X visualization module."""
import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from driftx.viz.plotter import (
    load_results,
    plot_accuracy_over_time,
    plot_cost_accuracy_tradeoff,
    plot_retrain_timeline,
    plot_dis_components,
    generate_all_figures,
)


@pytest.fixture
def mock_results_dir(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic results dataframe
    rows = []
    policies = ['p0_never', 'p1_fixed', 'p5_dis_fused']
    names = ['Never-Retrain', 'Fixed-Schedule', 'DIS-Fused']
    
    for seed in [42, 43]:
        for p_idx, (p_id, p_name) in enumerate(zip(policies, names)):
            for w in range(5):
                rows.append({
                    "policy": p_id,
                    "policy_name": p_name,
                    "seed": seed,
                    "window_id": w,
                    "accuracy": 0.8 + 0.05 * p_idx - 0.01 * w,
                    "f1": 0.78 + 0.05 * p_idx - 0.01 * w,
                    "retrained": (w % 2 == 0) if p_id != 'p0_never' else (w == 0),
                    "retrain_reason": "Scheduled" if w % 2 == 0 else "None",
                    "cost_seconds": 0.05 if (w % 2 == 0 or w == 0) else 0.0,
                    "cumulative_cost": 0.05 * (w + 1),
                    "cumulative_retrains": (w // 2 + 1) if p_id != 'p0_never' else 1,
                    "stat_drift_score": 0.1 * w,
                    "shap_magnitude_score": 0.05 * w,
                    "shap_rank_change_score": 0.02 * w,
                    "dis": 0.08 * w,
                    "threshold": 0.3,
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "experiment_results.csv", index=False)
    return results_dir


class TestVizPlotter:

    def test_load_results(self, mock_results_dir):
        df = load_results(mock_results_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30

    def test_plot_accuracy_over_time(self, mock_results_dir):
        df = load_results(mock_results_dir)
        out_dir = mock_results_dir / "figures"
        plot_accuracy_over_time(df, output_dir=out_dir, metric="accuracy")
        
        assert (out_dir / "accuracy_over_time.png").exists()
        assert (out_dir / "accuracy_over_time.pdf").exists()

    def test_plot_cost_accuracy_tradeoff(self, mock_results_dir):
        df = load_results(mock_results_dir)
        out_dir = mock_results_dir / "figures"
        plot_cost_accuracy_tradeoff(df, output_dir=out_dir)
        
        assert (out_dir / "cost_accuracy_tradeoff.png").exists()
        assert (out_dir / "cost_accuracy_tradeoff.pdf").exists()
        assert (out_dir / "policy_summary.csv").exists()

    def test_plot_retrain_timeline(self, mock_results_dir):
        df = load_results(mock_results_dir)
        out_dir = mock_results_dir / "figures"
        plot_retrain_timeline(df, output_dir=out_dir)
        
        assert (out_dir / "retrain_timeline.png").exists()

    def test_plot_dis_components(self, mock_results_dir):
        df = load_results(mock_results_dir)
        out_dir = mock_results_dir / "figures"
        plot_dis_components(df, output_dir=out_dir)
        
        assert (out_dir / "dis_components.png").exists()

    def test_generate_all_figures(self, mock_results_dir):
        generate_all_figures(mock_results_dir)
        out_dir = mock_results_dir / "figures"
        
        assert (out_dir / "accuracy_over_time.png").exists()
        assert (out_dir / "f1_over_time.png").exists()
        assert (out_dir / "cost_accuracy_tradeoff.png").exists()
        assert (out_dir / "retrain_timeline.png").exists()
        assert (out_dir / "dis_components.png").exists()
