# SwitchRank — Experimental Evaluation & Final Report

## Executive Summary
SwitchRank is an evidence-driven machine learning system for cross-catalog product record matching, supplier-catalog normalization, probability calibration, hard-negative handling, and selective human-review routing (`MATCH`, `REVIEW`, `NON_MATCH`).

Evaluated on the Web Data Commons (WDC) `80cc20rnd` multi-dimensional benchmark (80% corner cases) and tested for zero-shot domain shift on FDA 510(k) medical device catalog resolution.

---

## 1. Reconciled Ablation & Model Comparison Table

| Pipeline Stage | 000un F1 | 050un F1 | 100un F1 | Precision | Recall | Hard-Negative FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Raw text / RapidFuzz (E0)** | 0.2804 | 0.3080 | 0.2982 | 0.1724 | 0.7500 | 0.5973 |
| **2. Weighted Rules Matcher (E1)** | 0.3881 | 0.4148 | 0.4205 | 0.2504 | 0.8620 | 0.4267 |
| **3. Fellegi–Sunter Linkage (E2)** | **0.3955** | **0.4252** | **0.4319** | 0.2618 | 0.8080 | 0.3797 |
| **4. Supervised LightGBM (E3)** | 0.3817 | 0.4105 | 0.4110 | 0.2513 | 0.7940 | 0.3933 |
| **5. LightGBM + Hard-Negative Mining** | 0.3868 | 0.4208 | 0.4214 | 0.2636 | 0.7260 | **0.3377** |
| **6. Selected Matcher + Calibration & Policy** | 0.3868 | 0.4208 | 0.4214 | **98.34%** | **10.71%** | **1.66%** |

*All metrics independently recomputed from data and stored in `reports/final_metrics.json`.*

---

## 2. Key Empirical Findings & Answers to Research Questions

### RQ1: String Dissimilarity Baseline (E0)
Simple RapidFuzz token set ratio suffers a **59.73% False Positive Rate on hard negatives**, yielding an overall F1 of 0.2804 on `000un` test data.

### RQ2: Candidate Blocking Pair Reduction & Known Limitation
Multi-pass candidate blocking (Brand + Prefix + Sorted Neighborhood) generates **9,352 candidate pairs** out of **499,500 possible comparisons** (**98.13% pair reduction ratio**) while preserving **70.80% blocking recall** (354 / 500 ground-truth matches retained).
> **Known Limitation**: While multi-pass blocking eliminates 98.13% of uninformative candidate pairs, a blocking recall of 70.80% means 29.20% of true matches are filtered out prior to scoring. Production systems require higher-recall candidate generation.

### RQ3 & RQ4: Fellegi–Sunter vs LightGBM Trade-Off
- **Fellegi–Sunter Linkage (E2)** achieves the highest overall matching accuracy (**0.3955 F1** on `000un`, **0.4252 F1** on `050un`, and **0.4319 F1** on unseen `100un` test data) while providing transparent log-likelihood field agreement weights ($W = \log_2 \frac{m}{u}$).
- **LightGBM + Hard-Negative Mining (E3)** minimizes dangerous hard-negative false matches, achieving the lowest hard-negative FPR (**33.77%**).
- **Decision**: The project treats both approaches as complementary evidence about overall matching accuracy versus high-risk false-match behavior.

### Direct Causal Effect of Hard-Negative Sample Weighting
Comparing standard LightGBM directly against $3.0\times$ hard-negative sample-weighted LightGBM on `test_000un`:
- Standard LightGBM Hard-Neg FPR: **39.33%**
- Hard-Negative Weighted LightGBM Hard-Neg FPR: **33.77%**
- **Direct Causal Intervention Effect**: **-5.57 percentage points** (**-14.15% relative** FPR reduction).

### RQ7 & RQ8: Probability Calibration & Selective Decision Policy
Evaluating calibration metrics on held-out `val_policy` (fitted strictly on `val_calib`), Isotonic Regression achieved the lowest Expected Calibration Error (**ECE = 0.0290** vs uncalibrated **0.0798** and Platt **0.0517**). Under the selective decision policy tuned for target 99% precision, the engine achieves **98.34% auto-match precision** at **10.71% coverage**, routing **89.29%** of ambiguous pairs to `REVIEW`.

### RQ9: Healthcare Domain-Shift & Abstention Experiment (FDA 510(k) Clearances)
Evaluated on FDA 510(k) Medical Device Clearances Database (`product_code: FMF`):
- Zero-shot transfer from the WDC product domain to FDA medical-device records produced substantial distribution shift.
- The frozen selective policy abstained on nearly all cases (1,199 / 1,200 cases routed to `REVIEW`, **99.92% review rate**), demonstrating that healthcare deployment would require domain-specific labeled matching data rather than relying on zero-shot transfer.
