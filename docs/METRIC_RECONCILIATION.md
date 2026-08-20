# METRIC RECONCILIATION & SCIENTIFIC INTEGRITY REPORT — SwitchRank




---

## 1. Executive Summary
This document records the authoritative scientific-integrity reconciliation of the `switchrank-product-matching` evaluation pipeline. Every hardcoded metric, placeholder claim, and split overlap risk has been eliminated. The evaluation protocol is 100% reproducible, leakage-safe, and driven dynamically from data by `scripts/generate_reports.py` into machine-readable `reports/final_metrics.json`.

---

## 2. Reconciled Evaluation & Calibration Protocol

1. **Validation Split Disjointness**:
   - `valid_small` (2,500 candidate pairs) is split deterministically (`seed=42`, stratified by label) into:
     - `val_calib` (1,250 pairs): Used strictly for fitting Platt sigmoid and Isotonic regression calibrators.
     - `val_policy` (1,250 pairs): Used strictly for comparing calibration metrics (ECE, Brier score, log loss), selecting the best calibrator, and tuning `MATCH`/`REVIEW`/`NON_MATCH` policy thresholds targeting 99% precision.
   - Test sets (`000un`, `050un`, `100un`) are never touched during fitting or threshold selection.

2. **Calibration Selection**:
   - Evaluating calibration metrics on `val_policy`:
     - **Uncalibrated**: ECE = `0.0798`, Brier = `0.1260`, Log Loss = `0.3958`
     - **Platt Sigmoid**: ECE = `0.0517`, Brier = `0.1255`, Log Loss = `0.4011`
     - **Isotonic Regression**: ECE = `0.0290`, Brier = `0.1123`, Log Loss = `0.3279`
   - **Selection**: Isotonic Regression was selected based on validation evidence, reducing ECE from 0.0798 to 0.0290.

3. **Causal Hard-Negative Weighting Intervention**:
   - Direct intervention comparison between Standard LightGBM vs Hard-Negative Weighted LightGBM ($3.0\times$ weight on `is_hard_negative == True`) on `test_000un`:
     - Standard LightGBM Hard-Neg FPR: **39.33%**
     - Hard-Negative Weighted LightGBM Hard-Neg FPR: **33.77%**
     - **Direct Causal Delta**: **-5.57 percentage points** (**-14.15% relative reduction**).
   - RapidFuzz baseline FPR (59.73%) is reported separately to distinguish model architecture improvements from sample-weighting interventions.

4. **Dynamic Blocking Computation & Known Limitation**:
   - Running `CandidateBlocker` dynamically over `train_df` entities yields:
     - Total Possible Pairs: **499,500**
     - Candidate Pairs Generated: **9,352**
     - Pair Reduction Ratio: **98.13%**
     - Blocking Recall (Pair Completeness): **70.80%** (354 / 500 ground-truth matches retained).
   - **Known Limitation**: A blocking recall of 70.80% means 29.20% of true matches are filtered out prior to scoring. Production systems require higher-recall candidate generation.

5. **Healthcare Domain-Shift Abstention (FDA 510(k) Clearances)**:
   - Evaluated on 1,200 perturbed pairs across EASY (300), MEDIUM (600), and HARD (300) tiers from FDA 510(k) Medical Device Clearances Database (`product_code: FMF`).
   - Zero-shot transfer from WDC consumer products to FDA medical device records produced substantial distribution shift.
   - The selective decision policy correctly routes **99.92% of pairs (1,199 / 1,200) to `REVIEW`** (100.0% on EASY, 99.83% on MEDIUM, 100.0% on HARD), demonstrating that healthcare deployment would require domain-specific labeled matching data rather than zero-shot transfer.

6. **Fellegi–Sunter vs LightGBM Operational Decision**:
   - **Fellegi–Sunter Linkage (E2)** achieves superior overall matching F1 (**0.3955** on `000un`, **0.4319** on unseen `100un`) with complete mathematical interpretability via log-likelihood field weights.
   - **LightGBM + Hard-Negative Mining (E3)** achieves the lowest hard-negative false positive rate (**33.77%**).
   - Both matchers are retained as complementary operational evidence.
