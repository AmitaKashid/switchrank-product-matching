# DECISION LOG — SwitchRank Architecture & Modeling Choices

This document tracks all empirical architectural, feature engineering, and model choices. All metrics align with `reports/final_metrics.json`.

---

## Decision 1 — Benchmark Configuration & Variant

**Question**
Which WDC Products benchmark configuration and development split size should be selected to stress-test corner-case product matching while remaining computationally practical?

**Hypothesis**
The `80cc20rnd` configuration (80% corner cases) will provide realistic hard-negative pairs without overfitting to trivial dissimilarity.

**Experiment**
Inspected WDC benchmark variants (`20cc80rnd`, `50cc50rnd`, `80cc20rnd`) and train sizes (`train_small`, `train_medium`). Evaluated pair counts and difficulty distributions.

**Result**
`80cc20rnd` contains 80% hard negatives (near-identical product titles with different specifications or variants). `train_small` provides 2,500 candidate pairs, allowing rapid iterative training while maintaining high difficulty.

**Decision**
Retained `80cc20rnd` with `train_small`/`valid_small` splits as the primary development benchmark, evaluated against official `000un`, `050un`, and `100un` test sets.

---

## Decision 2 — Probabilistic Linkage vs Supervised LightGBM

**Question**
Does supervised pairwise classification materially outperform classical Fellegi–Sunter probabilistic record linkage?

**Hypothesis**
Supervised LightGBM will outperform Fellegi–Sunter due to non-linear feature interaction modeling.

**Experiment**
Compared Fellegi–Sunter record linkage (E2) against standard LightGBM (E3) on WDC `test_000un` and `test_100un`.

**Result**
Fellegi–Sunter achieved **0.3955 F1** on `test_000un` and **0.4319 F1** on unseen `test_100un`, whereas standard LightGBM achieved 0.3850 F1 and 0.4182 F1. LightGBM + Hard-Negative sample weighting achieved **0.3868 F1** on `test_000un` and **0.4214 F1** on `test_100un`, while reducing hard-negative false positive rate down to **33.77%**.

**Decision**
Retained both LightGBM (for hard-negative FPR minimization) and Fellegi–Sunter (for interpretable log-likelihood agreement weights).

---

## Decision 3 — Selective Decision Policy & Numeric Overrides

**Question**
Can the model enforce 99.0% auto-match precision while routing ambiguous or numeric spec conflicts to human review?

**Hypothesis**
Enforcing validation-tuned probability thresholds and numeric mismatch penalty rules will eliminate false positive matches on capacity/spec conflicts.

**Experiment**
Tuned match/non-match thresholds on validation data targeting 99% precision with isotonic calibration.

**Result**
Achieved **98.34% auto-match precision** at **10.69% auto-match coverage** on `test_000un`, routing **89.31%** of ambiguous pairs to `REVIEW`. Zero false positive matches occurred on numeric mismatch overrides.

**Decision**
Enforced isotonic probability calibration and selective decision policy with numeric contradiction override.
