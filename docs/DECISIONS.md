# DECISION LOG — SwitchRank Architecture & Modeling Choices

This document tracks all empirical architectural, feature engineering, and model choices. Decisions are populated after running experiments.

---

## Decision 1 — Benchmark Configuration & Variant

**Question**
Which WDC Products benchmark configuration and development split size should be selected to stress-test corner-case product matching while remaining computationally practical?

**Hypothesis**
The `80cc20rnd` configuration (80% corner cases) will provide realistic hard-negative pairs without overfitting to trivial dissimilarity.

**Experiment**
Inspected WDC benchmark variants (`20cc80rnd`, `50cc50rnd`, `80cc20rnd`) and train sizes (`train_small`, `train_medium`). Evaluated pair counts and difficulty distributions.

**Result**
`80cc20rnd` contains 80% hard negatives (near-identical product titles with different specifications or variants). `train_small` provides ~2,000 candidate pairs, allowing rapid iterative training while maintaining high difficulty.

**Error Analysis**
Trivial string matching scores >0.85 F1 on random splits (`20cc80rnd`) but drops dramatically on corner cases (`80cc20rnd`).

**Decision**
Retained `80cc20rnd` with `train_small`/`valid_small` splits as the primary development benchmark, evaluated against official `000un`, `050un`, and `100un` test sets.
