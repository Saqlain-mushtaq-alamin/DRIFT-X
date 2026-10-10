# Pre-Registered Hypothesis and Analysis Plan

## 1. Hypothesis
At a matched retrain budget, the explainability-fused policy **P5 (DIS-Fused)** achieves higher out-of-sample prequential accuracy than baseline policies:
- **P1 (Fixed Schedule)**,
- **P2 (Calibrated KS-Test / Statistical Only)**, and
- **P8 (Random Budget Control)**
on non-stationary data streams characterized by **mixed harmless drift** (covariate shifts in non-informative features) and **harmful drift** (concept shifts in informative features).

### Mechanism
- Statistical drift detection (P2) is sensitive to arbitrary covariate shift and triggers retraining on harmless feature fluctuations where the decision boundary has not degraded, incurring budget exhaustion and high retraining costs.
- Fixed retraining (P1) is oblivious to data dynamics and suffers degradation when changes occur out of phase with the schedule.
- Random retraining (P8) consumes budget uniformly at random without signal.
- P5 fuses statistical drift with attribution dynamics (SHAP magnitude and rank shifts), filtering out harmless covariate drift and triggering retraining primarily when drift impacts model attribution.

---

## 2. Primary Evaluation Metric
1. **Mean Accuracy at Matched Retrain Budget**: Out-of-sample prequential accuracy evaluated at a fixed operational budget constraint of **3–5 net retrains** across the evaluation stream.
2. **Area Under the Accuracy-vs-Retrains Operating-Point Curve (AUC-ROC/Pareto Front)**: Integrating prequential accuracy over the operating range of 1 to 10 retrains.

---

## 3. Statistical Testing Plan
- **Paired Wilcoxon Signed-Rank Test**: Evaluated across $N = 30$ independent synthetic evaluation streams (seeds 1000–1029) with randomized change-point positions, random informative feature subsets, and continuous harmless background noise.
- **95% Bootstrap Confidence Interval**: Computed with $B = 5,000$ resamples on the paired differences $\Delta = \text{Accuracy}(P_5) - \text{Accuracy}(P_{\text{baseline}})$.
- **Rejection Criterion**: Significance declared at $\alpha = 0.05$ (two-sided).

---

## 4. Analysis and Disclosure Plan
- **Full Reporting**: Every stream will be reported in tabular and summary formats, including individual stream trajectories and any stream regimes where P5 underperforms P1, P2, P6, or standard streaming baselines.
- **Regime Breakdown**: Separate evaluations will be conducted on:
  1. *Mixed drift streams* (the target hypothesis regime: harmless covariate drift + abrupt concept shifts).
  2. *Gradual drift streams* (continuous drift across all features).
  3. *Pure concept drift streams* (decision boundary shifts with identical covariate marginal distributions).
- **Negative Result Commitment**: If P5 fails to achieve statistically significant gains over calibrated KS or P1 at matched budgets, the paper will transparently report this as an empirical finding: explanation gating provides diagnostic localization but does not improve trigger Pareto efficiency over calibrated statistical detectors.
