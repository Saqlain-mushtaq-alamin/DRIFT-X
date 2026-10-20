# Pre-Registered Hypothesis and Analysis Plan (v3)

**Registration Date**: October 20, 2026  
**Target Configuration**: `configs/frozen.yaml`  
**Configuration SHA-256 Hash**: `28d0339c240d`  
**Status**: FROZEN / PRE-REGISTERED  

---

## 1. Provenance and Seed Partition Protocol

To prevent overfitting, data peeking, and confirmation bias, evaluation streams are partitioned strictly into disjoint sets:

1. **Tuning and Calibration Seeds (Seeds 1–10)**:
   - Used exclusively for detector calibration, grid bounds determination (e.g. ADWIN sweep range, SHAP rank threshold adjustments), bug fixes, and pipeline smoke tests.
   - No confirmatory claims are derived from seeds 1–10.

2. **Exploratory Development Seeds (Seeds 1000–1029)**:
   - Generated during initial pipeline development and debugging. Designated as `_exploratory`.

3. **Confirmatory Evaluation Seeds v2 (Seeds 2000–2029)**:
   - Evaluated under HYPOTHESIS_v2.

4. **Extended Confirmatory Seeds (Seeds 3000–3029)**:
   - Evaluated under extended validation protocol. Treated as spent.

5. **Confirmatory Evaluation Seeds v3 (Seeds 4000–4029)**:
   - $N = 30$ completely fresh, unseen, independent synthetic streams generated across evaluation regimes, specifically `mixed_decoupled`.
   - Neither the model architecture, feature weighting algorithms, threshold grids, nor hyperparameters in `configs/frozen.yaml` will be adjusted after observing confirmatory seeds.
   - Confirmatory execution requires `--confirmatory --seed-start 4000 --n-streams 30` and a clean git working tree (`git_dirty: False`).

---

## 2. Generator Artifact Resolution & Decoupled Regime

### Investigation of Change-Point Artifact in `mixed`
In the original `mixed` synthetic stream generator, when a concept change occurs at window $w \in CP$, a new set of informative features is drawn from the feature pool. The background harmless covariate drift is applied every window to all non-informative features ($\mathcal{F} \setminus \mathcal{F}_{\text{informative}}$). Consequently, features that were informative prior to the change point ($w-1$) but are non-informative at $w$ immediately begin to undergo covariate drift.

Because the pre-change model's SHAP importance weights are highest on exactly those formerly informative features, importance-weighted drift detectors (such as **P7 Importance-Weighted KS** and **P5-wstat-only**) observe strong distribution shift on high-importance features immediately at the change point. This allowed label-free detectors to achieve artificially high recall ($\sim 93\%$–$99\%$) on "invisible" change points (pure concept shifts without injected feature covariate drift), whereas in `abrupt_only` (which lacks harmless drift), invisible change point recall was near chance ($\sim 25\%$).

### Decoupled Regime Specification (`mixed_decoupled`)
To eliminate this artifact and isolate true concept-drift detection from harmless-drift bleed-through, we introduce the decoupled regime:
- **Harmful Feature Pool**: Features $\mathcal{P}_h = \{0, 1, 2, 3, 4, 5\}$ (fixed). All informative features and concept coefficients are selected exclusively from $\mathcal{P}_h$.
- **Harmless Feature Pool**: Features $\mathcal{P}_d = \{6, 7, \dots, 14\}$ (fixed). Harmless background covariate drift is applied strictly to features in $\mathcal{P}_d$.
- **Invariance**: Concept shifts within $\mathcal{P}_h$ never alter which features experience covariate drift. Former informative features never receive harmless covariate drift, eliminating the spurious correlation between concept shifts and weighted KS triggers.

---

## 3. Formal Hypotheses

### Primary Hypothesis (H1: Matched-Budget Efficiency under Decoupled Drift)
On non-stationary streams exhibiting decoupled mixed drift (`mixed_decoupled`), we evaluate whether importance weighting and explanation fusion provide genuine matched-budget gains over unweighted statistical drift detection:
1. **P7 (Importance-Weighted KS) vs P2 (Calibrated KS Drift Only)**:
   - Evaluated on curve Area Under the Curve (AUC, $B \in [1, 10]$) and accuracy at budget $B = 4$ retrains.
   - Direction: P7 achieves superior AUC and acc@4 compared to P2.
2. **P5-wstat-only (SHAP-Weighted Calibrated KS) vs P2 (Calibrated KS Drift Only)**:
   - Evaluated on AUC and acc@4.
   - *Disclosure of Exploratory Status*: The direct comparison of P5-wstat-only against P2 was not pre-declared in HYPOTHESIS_v2; any confirmatory claim regarding P5-wstat-only vs P2 from this run is formally designated as exploratory / post-hoc verification.
3. **P5 (DIS-Fused) vs Baselines**:
   - P5 vs {P2 (Drift Only), P3 (SHAP Magnitude), P4 (SHAP Rank), P5-wstat-only, P5-mag-only}.
   - *Holm Family Refinement Disclosure*: `p5_stat_only` has been removed from the P5 primary family because it algebraically duplicates P2 (both evaluate unweighted calibrated KS drift), which previously inflated the Holm family count.

### Secondary Hypothesis (H2: Harmless Drift Selectivity)
Under pure harmless drift (`harmless_only`), feature-weighted and explainability-fused policies (**P7**, **P5-wstat-only**, **P5**) trigger significantly fewer false retrains than unweighted statistical testing (**P2**), maintaining baseline model performance without wasting budget.

### Secondary Hypothesis (H3: Real-World Generalization Beyond ELEC2)
To verify whether weighted-KS and DIS-fused gains generalize beyond synthetic streams and beyond datasets where retraining accuracy quickly saturates (e.g. ELEC2, where accuracy plateaus after $\sim 20$ retrains), policies are evaluated on an additional real-world concept-drift streaming benchmark.

---

## 4. Grid Extensions and Dropped Test Disclosures

### Grid Densification for Coverage Assurance
In prior evaluations under v2, P7 achieved an AUC coverage of 0.55 on certain regimes (below the pre-registered minimum threshold of $\text{MIN\_COV\_AUC} = 0.70$), which led to the automatic dropping of the P7 vs P2 AUC test. 

To resolve coverage gaps and ensure all primary tests satisfy the $\ge 0.70$ threshold without extrapolation:
- **P7 Grid**: Extended with dense operating thresholds in $[0.5, 3.0]$ (19 total evaluation points).
- **wstat Grid**: Assigned a dedicated threshold grid with dense points in $[0.5, 3.0]$ (20 total evaluation points).

### Disclosure of Dropped Tests and Family Size
In previous v2 runs, up to 46 tests per regime were dropped due to coverage filters ($\text{near@4} < 0.80$ or $\text{auc\_cov} < 0.70$). Because dropped tests were omitted prior to computing step-down Holm corrections, the actual corrections were applied over smaller families than originally declared. We transparently disclose this shrinkage; in v3, denser grids are designed to eliminate test dropping.

---

## 5. Statistical Testing Protocol

1. **Paired Wilcoxon Signed-Rank Test**:
   - Two-sided Wilcoxon signed-rank test on paired stream metrics ($N = 30$ independent streams per regime).
   - Significance threshold: $\alpha = 0.05$.
2. **Family Multiplicity Control**:
   - Holm-Bonferroni step-down correction applied within each declared family:
     - **Primary Family**:
       - P5 vs {P2, P3, P4, P5-wstat-only, P5-mag-only} (5 comparisons)
       - P7 vs {P2, P3, P5} (3 comparisons)
       - P5-wstat-only vs P2 (1 comparison, exploratory)
     - **Reference Family**:
       - P5 and P7 vs fixed schedules, random retraining, and supervised performance drop / loss detectors.
3. **95% Bootstrap Confidence Intervals**:
   - 5,000 bootstrap resamples per comparison for differences in AUC and acc@4.
4. **Reproducibility Guarantee**:
   - Git tree must be clean (`git_dirty: False`).
   - All results tagged with commit SHA and configuration hash (`28d0339c240d`).
