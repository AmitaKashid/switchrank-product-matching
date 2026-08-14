# SwitchRank — Experimental Evaluation & Final Report

## Executive Summary
SwitchRank is an evidence-driven machine learning system for cross-catalog product record matching, confidence calibration, and selective human-review routing. Evaluated on the Web Data Commons (WDC) `80cc20rnd` multi-dimensional benchmark (80% corner cases) and zero-shot transferred to FDA 510(k) medical device catalog resolution.

---

## 1. Reconciled Ablation & Model Comparison Table

| Pipeline Stage | 000un F1 | 100un F1 | Precision | Recall | Hard-Negative FPR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Raw text / RapidFuzz (E0)** | 0.2804 | 0.2982 | 0.1724 | 0.7500 | 0.5973 |
| **2. Weighted Rules Matcher (E1)** | 0.3881 | 0.4205 | 0.2504 | 0.8620 | 0.4267 |
| **3. Fellegi–Sunter Linkage (E2)** | **0.3955** | **0.4319** | 0.2618 | 0.8080 | 0.3797 |
| **4. Supervised LightGBM (E3)** | 0.3817 | 0.4110 | 0.2513 | 0.7940 | 0.3933 |
| **5. LightGBM + Hard-Negative Mining** | 0.3868 | 0.4214 | 0.2636 | 0.7260 | **0.3377** |
| **6. Selected Matcher + Calibration & Policy** | 0.3868 | 0.4214 | **98.34%** | **10.71%** | **1.66%** |

---

## 2. Key Empirical Findings & Answers to Research Questions

### RQ1: String Dissimilarity Baseline (E0)
Simple RapidFuzz token set ratio suffers a **59.73% False Positive Rate on hard negatives**, yielding an overall F1 of 0.2804.

### RQ2: Candidate Blocking Pair Reduction
Multi-pass blocking (Brand + Prefix + Sorted Neighborhood) generates **9352 candidate pairs** out of 499500 possible comparisons (**98.13% pair reduction ratio**) while preserving **70.80% blocking recall**.

### RQ3 & RQ4: Fellegi–Sunter vs LightGBM Trade-Off
- **Fellegi–Sunter Linkage (E2)** achieves superior overall matching quality (**0.3955 F1** on `000un`, **0.4319 F1** on unseen `100un`) while being fully interpretable via explicit log-likelihood agreement weights ($W = \log_2 \frac{m}{u}$).
- **LightGBM + Hard-Negative Mining (E3)** minimizes dangerous hard-negative false matches, achieving the lowest hard-negative FPR (**33.77%**).

### Causal Effect of Hard-Negative Sample Weighting
Comparing standard LightGBM directly against hard-negative weighted LightGBM ($3.0\times$ sample weight):
- Hard-Negative FPR reduced from **39.33% down to 33.77%** (a **-5.57 percentage point** or **-14.15% relative** reduction).

### RQ7 & RQ8: Calibration & Selective Decision Policy
Isotonic regression calibration achieved the lowest calibration error on `val_policy` (ECE: **0.0290** vs uncalibrated **0.0798**). Under the selective decision policy tuned for target 99% precision, the engine achieves **98.34% auto-match precision** at **10.71% coverage**, routing **89.29%** of ambiguous pairs to human review.

### RQ9: Healthcare Domain-Shift & Abstention (FDA 510(k) Clearances)
Evaluated on FDA 510(k) Medical Device Clearances Database (`product_code: FMF`):
- Because medical device descriptions differ significantly from consumer e-commerce offers, zero-shot calibrated probabilities fall into the ambiguous threshold range.
- The selective policy correctly routes **99.83%+ to 100.0% of pairs to REVIEW** across EASY, MEDIUM, and HARD noise tiers, safely abstaining from automated medical device matching. Due to low auto-accepted sample sizes ($N \le 1$), transfer precision cannot be reliably estimated.
