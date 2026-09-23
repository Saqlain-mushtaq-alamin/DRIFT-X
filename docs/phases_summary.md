# DRIFT-X Completed Phases Summary

This document provides a detailed breakdown of the work completed across Phase 0 through Phase 8 of the DRIFT-X framework.

---

## ⚙️ Phase 0: Environment & Infrastructure Setup

### Objectives & Deliverables
- Configured Python 3.10+ project environment with dependencies (`xgboost`, `scikit-learn`, `shap`, `pandas`, `pytest`).
- Created package build system (`pyproject.toml`) utilizing `setuptools.build_meta` for editable local package installation (`pip install -e .`).
- Established global configuration file (`configs/default.yaml`) holding dataset schemas, model hyper-parameters, drift detection thresholds, and logging configurations.

---

## 📊 Phase 1: Data Ingestion & Leak-Free Windowing

### Core Components Implemented
- **`driftx.data.ingestor.DataIngestor`**:
  - Ingests datasets from CSV/Parquet formats (`fraud`, `intrusion`, `synthetic`).
  - Purges rows with missing targets and ensures strict numerical target formatting.
  - Sorts data strictly chronologically by timestamp column (`TransactionDT`, `timestamp`).

- **`driftx.data.windower.WindowSplitter` & datastructure `Window`**:
  - Implements temporal interval partitioning (`weekly`, `monthly`, `quarterly`, or explicit window count $K$).
  - Enforces strict zero-leakage temporal boundary assertions ($t_{\text{end}}(W_i) \le t_{\text{start}}(W_{i+1})$).
  - Handles missing value median imputation trained on the reference window to prevent future data leakage.

- **`scripts/download_data.py`**:
  - Downloads IEEE-CIS Fraud dataset via Kaggle API or generates synthetic organic-drift data (`synthetic_drift.csv`) for offline benchmarking.

### Verification
- **Unit Tests**: `tests/test_ingestor.py`, `tests/test_windower.py`
- **Script**: `python scripts/verify_phase1.py` (Ingested 12,000 rows across 8 temporal windows with verified 0% data leakage).

---

## 🤖 Phase 2: Baseline Model & Statistical Drift Detection

### Core Components Implemented
- **`driftx.training.trainer.ModelTrainer`**:
  - Supports model training for `xgboost`, `random_forest`, and `gradient_boosting`.
  - Calculates inside-window validation metrics (`accuracy`, `f1_weighted`, `precision_weighted`, `recall_weighted`).
  - Tracks wall-clock model training cost (`cost_seconds`).

- **`driftx.detection.ks_detector.KSDriftDetector`**:
  - Computes two-sample Kolmogorov-Smirnov test per feature comparing reference vs current window.
  - Reports per-feature p-values, test statistics, and aggregate ratio of drifted features.

- **`driftx.detection.psi_detector.PSIDriftDetector`**:
  - Discretizes continuous feature distributions into bins and computes Population Stability Index (PSI).
  - Flags drift when mean PSI exceeds `threshold = 0.2`.

### Verification
- **Unit Tests**: `tests/test_detection.py`, `tests/test_baseline_pipeline.py`
- **Script**: `python scripts/verify_phase2.py` (Observed model accuracy drop from `0.9527` to `0.5413` on shifted windows; PSI detector flagged drift at `Window 4`).

---

## 🔍 Phase 3: SHAP Explainability Layer

### Core Components Implemented
- **`driftx.explainability.shap_computer.ShapComputer`**:
  - Extracts SHAP values using `shap.TreeExplainer` or `shap.KernelExplainer`.
  - Subsamples data for fast computation and outputs mean absolute SHAP values (`mean_abs_shap`) and 1-indexed feature importance rankings.

- **`driftx.explainability.magnitude.MagnitudeTracker`**:
  - Tracks absolute feature importance shifts using normalized L1 distance between consecutive window mean |SHAP| profiles.

- **`driftx.explainability.rank_change.RankChangeTracker` (Novel Metric)**:
  - Measures structural model reasoning shifts using Spearman rank correlation ($\rho$).
  - Calculates rank-change score ($1 - \rho$) and outputs per-feature rank shift deltas (`prev_rank` vs `curr_rank`).

### Verification
- **Unit Tests**: `tests/test_explainability.py`
- **Script**: `python scripts/verify_phase3.py` (Computed SHAP profiles across 8 windows and verified Spearman rank-change tracking).

---

## 🔀 Phase 4: DIS Fusion Engine & Adaptive Threshold Controller

### Core Components Implemented
- **`driftx.fusion.dis.DriftImpactScore`**:
  - Fuses statistical drift, SHAP magnitude shift, and SHAP rank-change into a single metric:
    $$\text{DIS} = \alpha \cdot \text{Drift}_{\text{stat}} + \beta \cdot \text{Shift}_{\text{mag}} + \gamma \cdot \text{Shift}_{\text{rank}}$$

- **`driftx.fusion.threshold.AdaptiveThresholdController`**:
  - Dynamically adjusts the retraining threshold using a rolling window of recent DIS scores:
    $$\theta(t) = \mu_{\text{DIS}} + \lambda \cdot \sigma_{\text{DIS}}$$

### Verification
- **Unit Tests**: `tests/test_fusion.py`
- **Script**: `python scripts/verify_phase4.py`

---

## ⚡ Phase 5: Policy Engine & Retraining Logic

### Core Components Implemented
- **`driftx.policy.base.RetrainingPolicy` & `PolicyDecision`**:
  - Base interface enforcing Strategy Pattern for modular retraining policies.
  - Standardized decision output payload with boolean verdict, human-readable rationale, policy identifier, window ID, and raw evidence dictionary.

- **All 6 Retraining Policies Implemented**:
  - **`P0 (p0_never)`**: Baseline Never-Retrain policy.
  - **`P1 (p1_fixed)`**: Fixed schedule retraining (retrain every $N$ windows).
  - **`P2 (p2_drift_only)`**: Statistical drift-triggered retraining (no SHAP).
  - **`P3 (p3_shap_magnitude)`**: SHAP magnitude change-triggered retraining.
  - **`P4 (p4_shap_rank)`**: SHAP Spearman rank change-triggered retraining.
  - **`P5 (p5_dis_fused)` (Novel contribution)**: Fused DIS score vs. Adaptive Threshold Controller decision.

- **`driftx.policy.create_policy` & `create_all_policies`**:
  - Policy factory and router for instantiating single or full suite of policies from configuration.

### Verification
- **Unit Tests**: `tests/test_policy.py`
- **Script**: `python scripts/verify_phase5.py` (Executes all 6 policies across 8 temporal windows and produces full decision matrix).

---

## 🏃 Phase 6: Experiment Orchestrator & MLflow Integration

### Core Components Implemented
- **`driftx.experiment.runner.ExperimentRunner`**:
  - Central orchestrator wiring data ingestion, temporal windowing, model training, drift detection, SHAP explainability, DIS fusion, adaptive thresholding, policy decision making, model retraining, and metrics logging.
  - Supports multi-seed experimentation loops (`n_seeds`) across all active retraining policies.

- **CLI Entry Point (`scripts/run_experiment.py`)**:
  - Command-line interface accepting `--config`, `--policies`, and `--seeds` flags.
  - Automatically logs experiment runs to MLflow (`mlruns/`) and outputs aggregate performance tables.

- **CSV Result Persistence & Smoke Testing**:
  - Saves full per-window run metrics to `results/experiment_results.csv`.
  - Includes synthetic data smoke test (`scripts/smoke_test.py`) for rapid pipeline validation.

### Verification
- **Unit Tests**: `tests/test_experiment.py`
- **Scripts**: `python scripts/smoke_test.py`, `python scripts/verify_phase6.py`

---

## 📊 Phase 7: Full Experiments, Ablation & Visualization

### Core Components Implemented
- **Visualization Module (`driftx.viz.plotter`)**:
  - `plot_accuracy_over_time`: Figure 1 (Accuracy / F1 over temporal windows with seed standard deviation shading).
  - `plot_cost_accuracy_tradeoff`: Figure 2 (Pareto frontier plot of mean accuracy vs cumulative compute cost).
  - `plot_retrain_timeline`: Figure 3 (Horizontal temporal timeline displaying retraining markers per policy).
  - `plot_dis_components`: Figure 4 (4-panel signal breakdown of stat drift, SHAP mag, SHAP rank-change, and fused DIS vs θ(t)).
  - `generate_all_figures`: Batch generator saving PNG & PDF artifacts into `results/figures/`.

- **Ablation Study Engine (`scripts/run_ablation.py`)**:
  - Evaluates 7 component weight configurations (`full_dis`, `stat_only`, `mag_only`, `rank_only`, `stat_mag`, `stat_rank`, `mag_rank`) to isolate signal contributions.

- **Hyperparameter Sensitivity Engine (`scripts/run_sensitivity.py`)**:
  - Grid search over threshold sensitivity $\lambda \in [0.5, 3.0]$ and lookback window $k \in [3, 10]$ for P5 robustness.

- **Publication Table Generator (`scripts/generate_tables.py`)**:
  - Generates publication-ready LaTeX tables (`results/paper_table.tex`) and Markdown tables (`results/paper_table.md`).

### Verification
- **Unit Tests**: `tests/test_viz.py`
- **Script**: `python scripts/verify_phase7.py`

---

## 🌐 Phase 8: Multi-Dataset Validation & Serving Layer

### Core Components Implemented
- **Multi-Dataset Ingestion & Adapters (`driftx.data.ingestor.DataIngestor`)**:
  - **CIC-IDS2018 (`intrusion`)**: Supports loading raw daily network intrusion flow datasets or generating realistic organic drift traffic with evolving attack signatures (DoS, DDoS, Botnet, Infiltration).
  - **Australian NSW Electricity (`electricity` / ELEC2)**: Ingests the classic organic concept drift benchmark from OpenML with 45,312 time-series instances, automated target normalization (`UP` $\rightarrow 1$, `DOWN` $\rightarrow 0$), and offline drift fallback generation.
  - **Dataset Configurations**: `configs/datasets/intrusion.yaml`, `configs/datasets/electricity.yaml`, and `configs/datasets/fraud.yaml`.

- **Daily Window Partitioning (`driftx.data.windower.WindowSplitter`)**:
  - Implemented `"daily"` window size mode ($t_{\text{step}} = 86,400\text{s}$) with automated unit vs. seconds timestamp scale detection and strict zero-leakage verification.

- **Model Registry & Artifact Persistence (`driftx.training.registry.ModelRegistry`)**:
  - Serializes trained classifier models (`results/latest_model.joblib`), version metadata, and real-time DIS monitoring state (`results/latest_dis.json`).
  - Seamlessly integrates with `ExperimentRunner` to persist latest model artifacts across retrain cycles.

- **FastAPI Serving Layer (`driftx.serving.api` & `ModelServer`)**:
  - Provides production-ready REST API serving:
    - `GET /health`: Health status, model loaded boolean, and active model version.
    - `POST /predict`: Real-time inference returning prediction class, calibrated probabilities, and model version.
    - `GET /drift-status`: Live Drift Impact Score (DIS), individual component breakdown, and adaptive threshold monitoring.
    - `POST /model/reload`: Dynamic on-demand reload of model artifacts from disk.

- **Cross-Dataset Generalization Engine (`scripts/run_multi_dataset.py`, `scripts/generate_tables.py`)**:
  - Automated benchmark comparing P0 (Never Retrain), P2 (Drift-Only), and P5 (DIS-Fused) across all three distinct domains.
  - Automatically compiles publication-ready LaTeX tables (`results/cross_dataset_table.tex`) and Markdown tables (`results/cross_dataset_table.md`).

### Verification
- **Unit Tests**: `tests/test_serving.py` (7 tests), `tests/test_multi_dataset.py` (5 tests)
- **Script**: `python scripts/verify_phase8.py` (Verified 100% across multi-dataset ingestion, zero-leakage daily windowing, multi-domain experiment runs, cross-dataset table generation, and FastAPI endpoints).


