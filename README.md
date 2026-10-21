# DRIFT-X: On the Difficulty of Evaluating Retraining Triggers Under Matched Budgets

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: TMLR](https://img.shields.io/badge/Paper-TMLR%20Submission-red.svg)](paper/paper.md)
[![Audit: Reproducible](https://img.shields.io/badge/Audit-CURRENT%20(Frozen%2028d0339c240d)-brightgreen.svg)](planning/results_summary.md)

> **Official Research Repository and Reproducibility Package** for the paper:  
> *"On the Difficulty of Evaluating Retraining Triggers Under Matched Budgets: A Methodological Study, Benchmark Artifacts, and Empirical Limits"* ([paper/paper.md](paper/paper.md)).

---

## 📖 Executive Summary

When deploying machine learning models on non-stationary data streams, a fundamental operational question is **when to retrain**. An extensive literature proposes selective retraining triggers based on two-sample statistical tests, feature attribution weighting (e.g., TreeSHAP), and multi-channel signal fusion.

This repository implements **DRIFT-X**, an empirical evaluation framework that subjects continual retraining triggers to four rigorous methodological controls:
1. **Matched Operating Budgets:** Dual-anchored continuous operating curves evaluated across dense exponential threshold sweeps to decouple trigger selectivity from brute-force compute frequency.
2. **Strict Test-Then-Train Architecture:** Out-of-sample prequential evaluation on every temporal window prior to trigger decisions and model retraining.
3. **Decoupled Benchmark Generator:** Eradication of the structural generator bleed-through artifact that caused synthetic concept shifts to masquerade as detectable input feature drift.
4. **Pre-Registered Multiplicity Control:** Family-wise error rate control via the Holm-Bonferroni step-down procedure on pre-declared test families (`HYPOTHESIS_v3.md`) across $N=30$ independent streams.

---

## 🔍 Core Scientific Discoveries

### 1. The Benchmark Generator Artifact & Empirical Control
In standard synthetic concept drift generators (e.g., rotating random hyperplanes), harmless covariate drift is continuously injected into all features outside the currently informative set. When a concept shift occurs, previously informative features $\mathcal{S}_\text{inf}$ become uninformative and immediately begin drifting at the change point. Because historical model SHAP attributions are highest on $\mathcal{S}_\text{inf}$, importance-weighted two-sample tests detect massive drift at the change point, creating the illusion of label-free concept detection (**93.2% recall** on pure concept shifts vs. **43.4% chance**).

When we eliminate this artifact via the **Decoupled Generator** (`scripts/random_streams.py`)—strictly partitioning concept dynamics to features 0–5 and harmless covariate drift to features 6–14—invisible concept shift recall collapses to chance level (**45.2% vs. 43.4%**).

### 2. No Detectable Advantage Among Label-Free Triggers
Across 30 independent decoupled streams ($N=30$, seeds 4000–4029) under pre-registered confirmatory evaluation:
- Paired differences between importance-weighted KS (P7), multi-channel DIS-fusion (P5), and unweighted KS (P2) are statistically indistinguishable ($p_\text{Holm} \ge 0.673$).
- The 95% bootstrap confidence interval for P5 vs P2 at budget 4 (Acc@4) is `[-0.0015, +0.0258]` and for Curve AUC is `[-0.0054, +0.0083]`. While the data cannot rule out true differences up to about $\pm 0.01$ on AUC and $\pm 0.02$ on Acc@4, the evidence fails to show superiority.
- No label-free trigger outperforms a naive periodic fixed schedule (P1) at matched budget ($p_\text{Holm} = 0.955$ on AUC, $1.000$ on Acc@4).

### 3. Supervised (Error-Monitoring) vs. Label-Free Performance
Under matched budgets at budget 4 (Acc@4):
- **Page-Hinkley (P13)** significantly outperforms label-free triggers by **+3.1% to +3.5%** in absolute predictive accuracy (Holm $p \le 0.006$).
- **ADWIN (P10)** does **not** achieve statistical significance at Acc@4 (+0.0257 vs P7, Holm $p = 0.145$; +0.0218 vs P5, Holm $p = 0.217$).
- On Curve AUC, both Page-Hinkley and ADWIN maintain statistically significant advantages across the full curve area (Holm $p \le 0.009$).

### 4. Real-World Stream Sensitivity Disparities
On physical and economic streaming benchmarks, trigger frequency is strongly policy-dependent:
- On **NOAA Weather (NEweather)**: Unweighted KS (P2) is conservative, retraining in **7 of 19 windows**, while importance-weighted P5 and P7 retrain in **18–19 of 19 windows**.
- On **ELEC2 (NSW Electricity Market)**: P2 retrains in **20 of 30 windows**, while P5 and P7 retrain in **29–30 of 30 windows**.
- The accuracy spread on Weather spans from **0.7525** (Never-Retrain) to **0.7706** (Weighted-KS)—a spread of 1.8 percentage points. The 0.0077 spread corresponds strictly to the Curve AUC difference between P5 (0.7587) and P2 (0.7664).

---

## 📊 Confirmatory Operating Curves (Artifact-Free Decoupled Streams, N=30)

| Policy ID | Policy Name | Supervision | Curve AUC | AUC Cov | Acc @ 3 | Acc @ 4 | Near@4 Cov | Acc @ 5 |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **P14** | Oracle | Full Info | **0.7520** | 0.30 | 0.7597 | 0.7643 | 0.43 | 0.7643 |
| **P13** | Page-Hinkley | **Label-Using** | **0.7423** | 0.71 | 0.7488 | 0.7495 | 0.93 | 0.7505 |
| **P10** | ADWIN | **Label-Using** | **0.7394** | 0.79 | 0.7384 | 0.7400 | 0.97 | 0.7457 |
| **P6** | Performance-Drop | **Label-Using** | **0.7374** | 0.68 | 0.7449 | 0.7354 | 0.90 | 0.7479 |
| **P9** | SHAP-Loss | **Label-Using** | **0.7221** | 0.77 | 0.7137 | 0.7222 | 0.97 | 0.7350 |
| **P5** | DIS-Fused | Label-Free | **0.7176** | 0.88 | 0.6981 | 0.7183 | 0.97 | 0.7319 |
| **P7** | Weighted-KS | Label-Free | **0.7164** | 0.75 | 0.7032 | 0.7144 | 0.97 | 0.7214 |
| **P5-stat**| stat-only | Label-Free | **0.7161** | 0.68 | 0.6915 | 0.7067 | 0.73 | 0.7222 |
| **P2** | Drift-Only (KS) | Label-Free | **0.7159** | 0.68 | 0.6915 | 0.7067 | 0.73 | 0.7222 |
| **P5-wstat**| wstat-only | Label-Free | **0.7152** | 0.79 | 0.6989 | 0.7094 | 0.77 | 0.7201 |
| **P1** | Fixed-Schedule | None | **0.7124** | 0.90 | 0.6995 | 0.7170 | 1.00 | 0.7228 |
| **P5-mag** | mag-only | Label-Free | **0.7060** | 0.45 | 0.6981 | 0.7072 | 0.73 | 0.7135 |
| **P3** | SHAP-Magnitude | Label-Free | **0.7060** | 0.46 | 0.6984 | 0.7077 | 0.73 | 0.7132 |
| **P8** | Random-Budget | None | **0.6926** | 0.85 | 0.6734 | 0.6853 | 0.97 | 0.7011 |
| **P4** | SHAP-Rank-Change | Label-Free | **0.6904** | 0.43 | 0.6789 | 0.6853 | 0.37 | 0.6928 |
| **P12** | KSWIN | Label-Free | **0.6769** | 0.35 | 0.6558 | 0.6663 | 0.50 | 0.6759 |

---

## 🛠️ Repository Architecture

```
DRIFT-X/
├── configs/                # Frozen experimental configs (SHA256: 28d0339c240d)
│   ├── default.yaml        # Standard defaults
│   ├── frozen.yaml         # Locked configuration hash
│   └── datasets/           # Dataset specifications (weather, electricity, fraud, intrusion)
├── driftx/                 # Core Python package
│   ├── data/               # Ingestion, schema validation, and test-then-train windower
│   ├── detection/          # KS-test, PSI, and statistical two-sample detectors
│   ├── explainability/     # Subsampled TreeSHAP, magnitude, and rank-change trackers
│   ├── fusion/             # DIS signal fusion and Adaptive Threshold Controller (ATC)
│   ├── policy/             # Strategy implementations for all 14 policies (P0 through P14)
│   ├── training/           # XGBoost trainer, evaluators, and ModelRegistry
│   └── serving/            # FastAPI real-time REST prediction and drift monitoring server
├── docs/                   # Methodological documentation
│   ├── overview.md         # System architecture & core design principles
│   ├── phases_summary.md   # Chronological development breakdown (Phases 0 through 10)
│   └── usage_guide.md      # CLI execution and configuration guide
├── paper/                  # Publication manuscript
│   └── paper.md            # Complete academic paper (TMLR format)
├── planning/               # Open science audit trail
│   ├── results_summary.md  # Dynamically generated comprehensive results and audits
│   └── architecture/       # Phase design documents
├── results/                # Versioned raw CSVs, statistical tests, and operating curves
├── scripts/                # Experiment execution, verification, and audit scripts
│   ├── run_budget_curves.py       # Dense threshold sweeps & paired Wilcoxon tests
│   ├── run_real_weather.py        # NOAA NEweather real stream benchmark
│   ├── run_null_stream.py         # False-alarm calibration on stationary null streams
│   ├── analyze_relaxed.py         # Coverage-relaxed exploratory comparisons
│   └── generate_results_summary.py# Automated provenance & reproducibility auditor
├── HYPOTHESIS_v3.md        # Pre-registered hypotheses and analysis plan
├── requirements-lock.txt   # Pinned dependencies (pip freeze)
└── paper.md                # Root link to publication manuscript
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Saqlain-mushtaq-alamin/DRIFT-X.git
cd DRIFT-X

# Install pinned dependencies
pip install -r requirements-lock.txt
pip install -e .
```

### 2. Run Test Suite
```bash
pytest -v
```

### 3. Replicate Confirmatory Benchmark on Decoupled Streams
```bash
python scripts/run_budget_curves.py --regime mixed_decoupled --seed-start 4000 --n-streams 30 --confirmatory
```

### 4. Replicate Real Weather Benchmark
```bash
python scripts/run_real_weather.py
```

### 5. Regenerate Provenance Audit and Results Summary
```bash
python scripts/generate_results_summary.py
```

The audit script inspects raw output CSVs, cross-checks commit hashes against the current tree, verifies the frozen configuration hash (`28d0339c240d`), and writes the updated report to `planning/results_summary.md`.

---

## 📜 Provenance & Open Science Audit

| Artifact / Stream | Data Nature | Size | Status | Config SHA256 | Commit |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Decoupled Confirmatory (Seeds 4000–4029)** | Synthetic (Decoupled) | 30 streams $\times$ 30,000 | **CURRENT** | `28d0339c240d` | `542c871af029` |
| **NOAA Weather Benchmark** | Real Physical | 18,159 rows (20 windows) | **CURRENT** | `28d0339c240d` | `be07733f68fe` |
| **ELEC2 Benchmark** | Real Economic | 45,312 rows (30 windows) | Exploratory / Appendix | `28d0339c240d` | `bf22cc43067e` |
| **Independent Gaussian Null** | Synthetic Null | 95 windows $\times$ 5 seeds | Verified ($0/95$ alarms) | `28d0339c240d` | `bf22cc43067e` |
| **Shuffled ELEC2 Null** | Shuffled Null | 95 windows $\times$ 5 seeds | Verified ($0/95$ alarms) | `28d0339c240d` | `bf22cc43067e` |

*Transparency Note on Pre-Registration Timeline:* Development occurred across four stream looks: Seeds 1000–1029 (exploratory round 1), Seeds 2000–2029 (exploratory round 2), Seeds 3000–3029 (confirmatory round 1 before finding the artifact), and Seeds 4000–4029 (confirmatory evaluation on decoupled streams). `HYPOTHESIS_v3.md` was committed to git (commit `53861c3`) before running seeds 4000–4029.

---

## 📝 Citation

```bibtex
@article{alamin2026driftx,
  title={On the Difficulty of Evaluating Retraining Triggers Under Matched Budgets: A Methodological Study, Benchmark Artifacts, and Empirical Limits},
  author={Alamin, Saqlain Mushtaq},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://github.com/Saqlain-mushtaq-alamin/DRIFT-X}
}
```

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
