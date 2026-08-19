# DRIFT-X Usage & Execution Guide

This guide explains how to execute unit tests, run phase verification scripts, and configure the DRIFT-X framework.

---

## 🚀 Quickstart

### 1. Installation & Environment Setup
Ensure Python 3.10+ is installed. Install the package in editable mode:
```powershell
pip install -e .
```

### 2. Running Unit & Integration Tests
Execute all unit tests across the entire test suite:
```powershell
pytest -v
```

Expected output:
```
tests/test_baseline_pipeline.py PASSED
tests/test_detection.py PASSED
tests/test_explainability.py PASSED
tests/test_ingestor.py PASSED
tests/test_windower.py PASSED
============================= 25 passed in ~5s =============================
```

---

## 🧪 Phase Verification Scripts

Run the phase verification scripts to validate specific pipeline components end-to-end:

### Phase 1: Data Ingestion & Windowing
Generates synthetic organic-drift data and verifies zero data leakage across temporal windows:
```powershell
python scripts/verify_phase1.py
```

### Phase 2: Model Training & Statistical Drift
Trains initial baseline XGBoost model and tracks downstream metric degradation and statistical drift (KS-test / PSI):
```powershell
python scripts/verify_phase2.py
```

### Phase 3: SHAP Explainability & Rank Change
Computes SHAP profiles and tracks feature importance magnitude shifts and Spearman rank shifts ($1 - \rho$):
```powershell
python scripts/verify_phase3.py
```

### Phase 4: DIS Fusion Engine & Adaptive Threshold Controller
Fuses statistical drift, SHAP magnitude shift, and SHAP rank-change into DIS and dynamically evaluates thresholds:
```powershell
python scripts/verify_phase4.py
```

### Phase 5: Policy Engine & Retraining Logic
Executes all 6 retraining policies (P0 through P5) across data windows and outputs the policy decision matrix:
```powershell
python scripts/verify_phase5.py
```

### Phase 6: Experiment Orchestrator & MLflow Integration
Runs full pipeline across seeds and active policies, logs to MLflow, and saves CSV results:
```powershell
# Run smoke test on synthetic data
python scripts/smoke_test.py

# Run Phase 6 verification
python scripts/verify_phase6.py

# Run CLI experiment runner (custom policies and seed count)
python scripts/run_experiment.py --policies p0_never p5_dis_fused --seeds 2
```

### Phase 7: Full Experiments, Ablation & Visualization
Generates paper figures, executes ablation and sensitivity studies, and formats LaTeX tables:
```powershell
# Run Phase 7 verification
python scripts/verify_phase7.py

# Run Ablation Study across 7 signal weight configurations
python scripts/run_ablation.py

# Run Sensitivity Analysis over λ and lookback k
python scripts/run_sensitivity.py

# Generate LaTeX & Markdown paper results tables
python scripts/generate_tables.py
```

---

## ⚙️ Configuration Parameters (`configs/default.yaml`)

You can customize dataset parameters, model selection, and drift thresholds in `configs/default.yaml`:

```yaml
data:
  dataset: "fraud"
  timestamp_col: "TransactionDT"
  target_col: "isFraud"
  window_size: "monthly"
  min_window_samples: 1000

model:
  type: "xgboost"      # Options: xgboost, random_forest, gradient_boosting
  params:
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1

detection:
  ks_alpha: 0.05
  psi_threshold: 0.2

explainability:
  shap_method: "tree"
  max_shap_samples: 500
  magnitude_threshold: 0.1
  rank_change_threshold: 0.3
```
