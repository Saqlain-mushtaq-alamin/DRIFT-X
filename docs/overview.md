# DRIFT-X: Technical Architecture & System Overview

DRIFT-X is an explainability-driven ML drift detection and adaptive model retraining framework. It combines statistical feature distribution drift detection (KS-test, PSI) with tree-based SHAP explainability metrics (L1 magnitude shift & Spearman rank correlation structural shift) to calculate a unified Drift Importance Score (DIS) and intelligently trigger model retraining.

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────────────────┐
                    │       Data Ingestion & Windowing     │
                    │  (DataIngestor, WindowSplitter)       │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │      Model Training & Evaluation     │
                    │   (ModelTrainer - XGBoost/RF/GB)     │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Detection Layer (3 Signals)                           │
│                                                                             │
│  ┌─────────────────────────┐ ┌──────────────────────┐ ┌──────────────────┐ │
│  │   Statistical Drift     │ │    SHAP Magnitude    │ │    SHAP Rank     │ │
│  │   (KS-Test & PSI)       │ │    (L1 Distance)     │ │   (Spearman ρ)   │ │
│  └────────────┬────────────┘ └──────────┬───────────┘ └────────┬─────────┘ │
└───────────────┼─────────────────────────┼──────────────────────┼───────────┘
                │                         │                      │
                └─────────────────────────┼──────────────────────┘
                                          │
                                          ▼
                    ┌──────────────────────────────────────┐
                    │          DIS Fusion Engine           │
                    │   DIS = α·Stat + β·ShapMag + γ·Rank  │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │     Adaptive Threshold Controller    │
                    │      θ(t) = μ(DIS) + λ · σ(DIS)      │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │       Phase 5: Policy Engine         │
                    │ (Full Retrain / Partial / Baseline)  │
                    └──────────────────────────────────────┘
```

---

## 📦 Phase Implementation Status

| Phase | Description | Key Modules | Status |
|---|---|---|---|
| **Phase 0** | Environment & Infrastructure Setup | `pyproject.toml`, dependencies, configs | ✅ Complete |
| **Phase 1** | Data Ingestion & Windowing | `DataIngestor`, `WindowSplitter`, `Window` | ✅ Complete |
| **Phase 2** | Baseline Model & Statistical Detection | `ModelTrainer`, `KSDriftDetector`, `PSIDriftDetector` | ✅ Complete |
| **Phase 3** | SHAP Explainability Layer | `ShapComputer`, `MagnitudeTracker`, `RankChangeTracker` | ✅ Complete |
| **Phase 4** | DIS Fusion & Adaptive Threshold Controller | `DriftImpactScore`, `AdaptiveThresholdController` | ✅ Complete |
| **Phase 5** | Policy Engine & Retraining Logic | `RetrainingPolicy`, `PolicyDecision`, `create_policy`, `P0`-`P5` | ✅ Complete |
| **Phase 7** | Full Experiments, Ablation & Visualization | `plotter.py`, `run_ablation.py`, `run_sensitivity.py`, paper tables | ✅ Complete |
| **Phase 8** | Multi-Dataset Validation | Cross-domain evaluation (intrusion, healthcare) & reporting | 🔄 Next |
