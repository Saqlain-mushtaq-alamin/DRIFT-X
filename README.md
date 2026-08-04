# DRIFT-X: Drift-Reactive Interpretable Feature-Tracking with eXplainability

A novel architecture and Python framework for cost-optimal, explanation-driven model maintenance under organic concept drift.

## Overview

DRIFT-X fuses statistical drift detection, SHAP feature-importance magnitude change, and SHAP rank-change into a single **Drift Impact Score (DIS)** with an **Adaptive Threshold Controller (ATC)**. It provides a standardized 6-policy benchmarking engine to evaluate retraining decisions on a cost-accuracy Pareto frontier.

### Key Components

- **Statistical Drift Detection**: Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI).
- **SHAP Tracking**: Subsampled tree/kernel SHAP with L1 magnitude drift and Spearman rank-correlation tracking.
- **DIS Signal Fusion**: Fused score $\text{DIS}(t) = \alpha \cdot \text{StatDrift}(t) + \beta \cdot \text{ShapMag}(t) + \gamma \cdot \text{ShapRank}(t)$.
- **Adaptive Thresholding**: Moving-window threshold $\theta(t) = \mu + \lambda \cdot \sigma$.
- **6-Policy Engine**: Benchmark P0 (Never), P1 (Fixed-Schedule), P2 (Drift-Only), P3 (SHAP-Magnitude), P4 (SHAP-Rank), and P5 (DIS-Fused).

## Package Structure

```
driftx/
├── data/           # Ingestion and windowing
├── detection/      # KS-test & PSI drift detectors
├── explainability/ # SHAP computation & rank/magnitude trackers
├── fusion/         # DIS computation & adaptive thresholding
├── policy/         # 6 retraining policies strategy pattern
├── training/       # Model training & registry
├── evaluation/     # Wall-clock cost logging & Pareto analysis
├── experiment/     # Orchestration & config management
├── serving/        # FastAPI model server
└── viz/            # Result plotting & monitoring dashboard
```

## Quickstart

```bash
# Install package in editable mode
pip install -e .

# Run experiment suite
python scripts/run_experiment.py --config configs/default.yaml
```

## License

MIT License. See [LICENSE](LICENSE) for details.
