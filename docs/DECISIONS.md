# DECISION LOG — SwitchRank Architecture & Modeling Choices

This document tracks all empirical architectural, feature engineering, and model choices. All metrics align with `reports/final_metrics.json`.

---

## Decision 1 — Candidate Blocking Strategy & Trade-Offs

**Question**
How should candidate pair generation be structured to prune $O(N^2)$ comparisons while evaluating blocking recall?

**Result**
Multi-pass candidate blocking (Brand + Prefix + Sorted Neighborhood) generates **9,352 candidate pairs** out of **499,500 possible comparisons** on `train_df`, achieving a **98.13% pair reduction ratio** and a **70.80% blocking recall** (354 / 500 ground-truth matches retained).

**Decision**
Document candidate blocking as an efficient $O(N \log N)$ baseline, while explicitly noting that a blocking recall of 70.80% is a known limitation requiring higher-recall candidate generation (e.g. ANN / vector embeddings) for production deployment.

---

## Decision 2 — Probabilistic Linkage vs Supervised LightGBM Trade-Off

**Question**
Which model architecture provides the optimal operational tradeoff between overall matching accuracy and hard-negative false-positive control?

**Result**
- **Fellegi–Sunter Probabilistic Linkage (E2)** achieves the highest overall matching accuracy (**0.3955 F1** on `000un` and **0.4319 F1** on unseen `100un` test data) while offering complete transparency via explicit log-likelihood agreement weights ($W = \log_2 \frac{m}{u}$).
- **Supervised LightGBM (E3)** paired with $3.0\times$ hard-negative sample weighting achieves the lowest hard-negative false positive rate (**33.77% Hard-Neg FPR** vs standard LightGBM's 39.33%).

**Decision**
Retain **both** matchers as complementary evidence about accuracy versus high-risk false-match behavior rather than declaring a single universal winner.

---

## Decision 3 — Leakage-Safe Probability Calibration & Selective Policy

**Question**
Which probability calibration method minimizes Expected Calibration Error on held-out validation data, and how should thresholds be selected?

**Result**
Split `valid_small` (2,500 pairs) into `val_calib` (1,250 pairs, for fitting calibrators) and `val_policy` (1,250 pairs, for evaluation and threshold tuning). On `val_policy`:
- **Uncalibrated**: ECE = 0.0798
- **Platt Sigmoid**: ECE = 0.0517
- **Isotonic Regression**: ECE = **0.0290**

Tuned on `val_policy` for target 99% precision, the selective policy ($\tau_{match} = 0.610$, $\tau_{non\_match} = 0.170$) achieved **98.34% auto-match precision** at **10.71% coverage** on `test_000un`, routing **89.29%** of ambiguous pairs to `REVIEW`.

**Decision**
Enforce Isotonic Regression calibration and validation-tuned selective decision thresholds with scalar numeric contradiction penalty overrides.
