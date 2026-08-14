# METRIC RECONCILIATION REPORT — SwitchRank

**Author**: Staff ML Engineer
**Date**: August 2026
**Repository**: `switchrank-product-matching`

---

## 1. Executive Summary
This document establishes **ONE authoritative evaluation protocol** and details the root causes of previous metric discrepancies across legacy experiment scripts and report drafts. All stale metrics have been removed. Every number in `README.md`, `reports/final_report.md`, `docs/FINAL_AUDIT.md`, and generated CSVs is now dynamically driven by `reports/final_metrics.json`.

---

## 2. Root Cause Analysis of Previous Metric Conflicts

| Discrepancy Item | Conflicting Values Reported | Root Cause Analysis | Reconciled Canonical Metric |
| :--- | :--- | :--- | :--- |
| **Fellegi–Sunter Linkage F1** | `0.3098` vs `0.3955` | `run_linkage.py` evaluated Fellegi–Sunter using un-normalized numeric comparison fields, whereas `generate_reports.py` evaluated Fellegi–Sunter on normalized attributes with unit standardizing (`10 mL` ↔ `10ml`), significantly boosting field agreement weights. | **`0.3955`** (Evaluated on normalized pairwise feature matrix). |
| **LightGBM + Hard-Negative F1** | `0.3377` vs `0.3868` | `run_hard_negatives.py` evaluated the classifier at a fixed probability threshold of `0.150`, whereas `generate_reports.py` evaluated at the optimal validation threshold (`best_t = 0.520`). | **`0.3868`** (Evaluated at validation-tuned threshold). |
| **Hard-Negative False Positive Rate (FPR)** | `0.3810` vs `0.3377` | `run_hard_negatives.py` reported FPR at fixed threshold `0.150` (`0.3810`), whereas `generate_reports.py` evaluated FPR at the validation-tuned decision threshold (`0.3377`). | **`0.3377`** (Evaluated at validation-tuned threshold). |
| **Hard-Negative FPR Reduction Claim** | `51.1% -> 38.1%` vs `61.6% -> 33.77%` | `51.1% -> 38.1%` compared unweighted LightGBM to weighted LightGBM at a fixed 0.15 threshold. `61.6% -> 33.77%` measures the baseline progression from raw RapidFuzz (E0) to hard-negative weighted LightGBM. | **61.60% (E0 Baseline) -> 33.77% (Final LightGBM)**; **36.93% (Unweighted LightGBM) -> 33.77% (Weighted LightGBM)**. Both are documented clearly. |
| **Selective Policy Precision / Coverage** | `98.34% / 10.69%` vs `99.0% / 48.2%` | `99.0% / 48.2%` was an uncalibrated target placeholder from early design drafts. `98.34% / 10.69%` is the actual empirical result on WDC `test_000un` when target precision is set to 99.0% on validation data. | **98.34% Auto-Match Precision at 10.69% Coverage** (Target 99% Precision); **94.20% Precision at 38.60% Coverage** (Target 95% Precision). |
| **Healthcare Dataset Identity** | `AccessGUDID` vs `FDA 510(k)` | `scripts/download_gudid.py` queries `api.fda.gov/device/510k.json?search=product_code:FMF`. AccessGUDID is NLM's UDI lookup portal, whereas openFDA 510(k) is the FDA Medical Device Clearance database. | **FDA 510(k) Device Clearances Database (Product Code FMF: Syringe, Piston)**. |

---

## 3. Authoritative Frozen Evaluation Protocol

Every model, metric, and figure is generated strictly under the following protocol:

- **Benchmark Dataset**: Web Data Commons (WDC) Products Benchmark (`80cc20rnd` variant, 80% hard pairs, 20% random pairs).
- **Split Sizes**:
  - `train`: 2,500 candidate pairs (`wdcproducts80cc20rnd000un_train_small.json.gz`)
  - `valid`: 2,500 candidate pairs (`wdcproducts80cc20rnd000un_valid_small.json.gz`)
  - `test_000un`: 4,500 candidate pairs (0% unseen entities)
  - `test_050un`: 4,500 candidate pairs (50% unseen entities)
  - `test_100un`: 4,500 candidate pairs (100% unseen entities)
- **Healthcare Transfer Dataset**: FDA 510(k) Device Clearances Database (`api.fda.gov/device/510k.json`, Product Code `FMF`). 749 canonical device identities perturbed into 1,200 evaluation pairs (`EASY`, `MEDIUM`, `HARD`).
- **Random Seed**: `42`
- **Validation Tuning**: All model decision thresholds and calibration functions are tuned exclusively on `valid` data to prevent test set leakage.
