# Pre-Registered Hypothesis and Analysis Plan (v2)

**Registration Date**: October 16, 2026  
**Target Configuration**: `configs/frozen.yaml`  
**Configuration SHA-256 Hash**: `28d0339c240d`  
**Status**: FROZEN / PRE-REGISTERED  

---

## 1. Provenance and Seed Partition Protocol

To prevent overfitting, data peeking, and confirmation bias, evaluation streams are partitioned strictly into three disjoint sets:

1. **Tuning and Calibration Seeds (Seeds 1–10)**:
   - Used exclusively for detector calibration, grid bounds determination (e.g. ADWIN sweep range, SHAP rank threshold adjustments), bug fixes (e.g. runner DIS calculation on stat-only variant), and pipeline smoke tests.
   - No confirmatory claims are derived from seeds 1–10.

2. **Exploratory Development Seeds (Seeds 1000–1029)**:
   - Generated during initial pipeline development and debugging.
   - Because code adjustments occurred during exploratory inspection of these seeds, they are treated as exploratory. Any artifact generated using seeds overlapping 1000–1029 is explicitly designated as `_exploratory`.

3. **Confirmatory Evaluation Seeds (Seeds 2000–2029)**:
   - $N = 30$ completely unseen, independent synthetic streams generated across all 5 evaluation regimes.
   - Neither the model architecture, feature weighting algorithms, threshold grids, nor hyperparameters in `configs/frozen.yaml` have been or will be adjusted after observing confirmatory seeds.
   - All confirmatory runs require `--confirmatory --seed-start 2000 --n-streams 30` and a clean git working tree (`git_dirty: False`).

---

## 2. Formal Hypotheses

### Primary Hypothesis (H1: Matched-Budget Efficiency under Mixed Drift)
On non-stationary data streams exhibiting **mixed drift** (background harmless covariate drift in non-informative features combined with abrupt concept shifts in informative features), the explainability-fused policy **P5 (DIS-Fused)** achieves higher out-of-sample prequential accuracy than unsupervised baselines:
- **P2 (Calibrated KS-Test / Statistical Drift Only)**
- **P7 (Importance-Weighted KS-Test)**
- **P1 (Fixed Schedule Retraining)**
- **P8 (Random Budget Allocation)**

evaluated at a matched retraining budget constraint of **$B = 4$ retrains** (with sensitivity bounds at $B = 3$ and $B = 5$).

### Secondary Hypothesis (H2: False Alarm Suppression under Harmless Drift)
On data streams exhibiting **harmless covariate shift only** (distribution shifts restricted to non-informative features with zero concept drift), **P5 (DIS-Fused)** triggers significantly fewer false retraining triggers than statistical-only detectors (**P2**, **P5-Stat-Only**), preserving retraining budget and avoiding model variance degradation without sacrificing baseline accuracy.

### Secondary Hypothesis (H3: Attribution Component Ablation)
Within the DIS fusion framework:
- Full DIS fusion (**P5**) outperforms or matches the unweighted statistical ablation (**P5-Stat-Only**) and magnitude-only ablation (**P5-Mag-Only**), confirming that both attribution magnitude and rank shifts provide complementary signals for isolating impactful drift.
- Importance weighting (**P5-wStat-Only**) substantially reduces false alarms relative to unweighted statistical testing (**P5-Stat-Only**) in the presence of harmless feature noise.

---

## 3. Evaluation Regimes

All policies will be evaluated across 5 distinct synthetic streaming regimes:

| Regime | Description | Change Points ($CP$) | Covariate Noise |
| :--- | :--- | :--- | :--- |
| **`mixed`** | Primary testbed: Informative feature concept shifts + background noise | $\sim 3$–$5$ abrupt CPs | Harmless feature shifts |
| **`gradual_only`** | Continuous gradual drift across features | $0$ abrupt CPs | Continuous linear drift |
| **`harmless_only`** | Covariate shifts restricted strictly to non-informative features | $0$ concept CPs | Active covariate shift |
| **`abrupt_only`** | Sudden concept drift in informative features with clean background | $\sim 3$–$5$ abrupt CPs | None |
| **`concept_only`** | Pure decision boundary shifts ($P(Y \mid X)$ changes, $P(X)$ stationary) | $\sim 3$–$5$ abrupt CPs | None |

---

## 4. Metrics and Operationalization

### 1. Primary Metric: Matched-Budget Accuracy at $B = 4$
- For each stream $s \in \{2000, \dots, 2029\}$ and policy $p$, the parameter grid is swept to generate a set of operating points $(R_{p,s}(\theta), A_{p,s}(\theta))$, where $R$ is the net number of retrains and $A$ is the out-of-sample prequential accuracy.
- Operating points are linearly interpolated to evaluate accuracy at exactly $B = 4$ retrains.
- **Coverage Filter**: To ensure statistical validity, a policy must achieve operating points that bracket or reach budget $B = 4$ on at least **80% of streams** ($\ge 24 / 30$ streams). Policies with $<80\%$ coverage are omitted from paired testing at $B = 4$ to prevent extrapolation artifacts.
- Sensitivity is assessed at $B = 3$ and $B = 5$.

### 2. Secondary Metric: Operating Curve Area (AUC)
- Normalized Area Under the Accuracy-vs-Retrains Curve integrated over $[0, 24]$ retrains using trapezoidal integration, bounded between chance-level accuracy and oracle accuracy.

### 3. Change-Point Metrics
- Precision, Recall, and Mean Detection Latency (in stream steps) evaluated against known ground-truth concept change-points, stratified by change-point visibility.

---

## 5. Statistical Testing Plan

1. **Paired Wilcoxon Signed-Rank Test**:
   - Test statistic: Two-sided Wilcoxon signed-rank test on paired stream accuracy differences:
     $$\Delta_s = \text{Acc}_s(P_5, B=4) - \text{Acc}_s(P_{\text{baseline}}, B=4)$$
   - Sample size: $N = 30$ independent streams.
   - Significance threshold: $\alpha = 0.05$.
   - Output reporting: Exact test statistic $W$, two-sided $p$-value, mean paired difference, and median paired difference.

2. **95% Bootstrap Confidence Intervals**:
   - $B = 5,000$ bootstrap resamples of the paired differences $\Delta_s$.
   - Non-parametric percentile interval reported as $[\Delta_{\text{low}}, \Delta_{\text{high}}]$.

3. **Label Availability Disclosures**:
   - In all summary tables and narrative findings, supervised detectors requiring streaming labels at test time (**P6 Performance-Drop**, **P9 SHAP-Loss**, **P10 ADWIN**, **P11 DDM**, **P13 Page-Hinkley**) must be explicitly demarcated from unsupervised / label-free detectors (**P2 Calibrated-KS**, **P5 DIS-Fused**, **P7 Weighted-KS**, **P12 KSWIN**).

---

## 6. Commitments to Open Reporting

1. **No Data Exclusion**: Results for all 30 streams will be tabulated and archived without filtering or cherry-picking.
2. **Transparent Negative Findings**: If P5 fails to achieve statistically significant superiority over P2 or P1 at matched budget $B=4$, this will be reported transparently: explanation gating provides diagnostic attribution and harmless-drift suppression, but does not outperform calibrated statistical testing under budget-matched execution.
3. **Reproducibility Guarantee**: Every confirmatory artifact must record:
   - `git_commit`: Clean Git HEAD SHA.
   - `git_dirty`: `False`.
   - `frozen_hash`: `28d0339c240d`.
