# FINAL AUDIT & REPAIR LOG — SwitchRank

**Auditor**: Senior Staff ML Engineer
**Date**: August 2026
**Repository**: `switchrank-product-matching`

---

## 1. Executive Summary & Audit Scope
This audit independently inspects, challenges, repairs, and validates the entire `switchrank-product-matching` repository. Every claim, baseline, metric, feature, calibration step, and domain transfer test was audited against code execution. All metrics are now driven by `reports/final_metrics.json`.

---

## 2. Audit Findings & Reconciled Resolutions

| ID | Category | Discovered Issue | Severity | Status / Resolution |
| :--- | :--- | :--- | :---: | :--- |
| **AUD-01** | **Metric Discrepancies & Stale Reports** | Previous reports had conflicting values (e.g. Fellegi–Sunter F1 0.3098 vs 0.3955; Hard-Neg FPR 38.10% vs 33.77%; Selective policy precision 99.0%/48.2% vs 98.34%/10.69%). | **HIGH** | **RECONCILED**: Established single canonical pipeline (`scripts/generate_reports.py`) generating authoritative `reports/final_metrics.json`. All reports now read from `final_metrics.json`. |
| **AUD-02** | **Healthcare Dataset Identity Naming** | The dataset was inconsistently referred to as AccessGUDID in text while `scripts/download_gudid.py` fetched from the openFDA 510(k) API endpoint (`api.fda.gov/device/510k.json?search=product_code:FMF`). | **HIGH** | **RECONCILED**: Clarified dataset name everywhere as **FDA 510(k) Medical Device Clearance Database (Product Code FMF: Syringe, Piston)**. |
| **AUD-03** | **Healthcare Domain Transfer Interpretation** | Previous text claimed high zero-shot resolution accuracy without distinguishing raw classification from selective policy abstention. | **HIGH** | **RECONCILED**: Updated report to document that calibrated probabilities on transferred medical devices fall in the ambiguous range, causing the selective policy to correctly route **89%+ to 99%+ of pairs to REVIEW**, preventing unsafe auto-matching. |
| **AUD-04** | **API Deprecation Warnings** | `src/switchrank/api/main.py` used deprecated Pydantic v1 `.dict()` calls and FastAPI `@app.on_event("startup")` handlers, producing deprecation warnings during pytest. | **MEDIUM** | **FIXED**: Refactored API to use FastAPI `lifespan` context manager and Pydantic v2 `.model_dump()` & `json_schema_extra`. All 15 unit tests pass cleanly. |
| **AUD-05** | **Data Leakage & Split Invariants** | Verified pair ID overlap between train, validation, and test sets. Verified feature extraction does not use target labels or pair IDs. | **CRITICAL** | **VERIFIED CLEAN**: `tests/test_leakage.py` passed with 0 pair overlap across all splits (`000un`, `050un`, `100un`). |
| **AUD-06** | **Licensing & Data Provenance** | Verified WDC Products provenance (Univ. of Mannheim CC BY 4.0) and FDA 510(k) (public domain). Confirmed GMDN terms and D-U-N-S numbers are excluded. | **HIGH** | **VERIFIED CLEAN**: Licensing explicitly documented in `docs/DATA_LICENSES.md` and `README.md` with required NLM/FDA attribution. Raw datasets excluded from Git. |

---

## 3. Reconciled Baseline & Ablation Table (`reports/final_metrics.json`)

| Pipeline Stage | Overall F1 | Precision | Recall | Hard-Negative FPR | Unseen Entity (100un) F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Raw text / RapidFuzz (E0)** | 0.2826 | 0.1728 | 0.7760 | 0.6160 | 0.2826 |
| **2. Weighted Rules Matcher (E1)** | 0.3855 | 0.2481 | 0.8640 | 0.4323 | 0.3132 |
| **3. Fellegi–Sunter Probabilistic Linkage (E2)** | **0.3955** | 0.2618 | 0.8080 | 0.3797 | **0.4319** |
| **4. Supervised LightGBM (E3)** | 0.3850 | 0.2569 | 0.7680 | 0.3693 | 0.4182 |
| **5. LightGBM + Hard-Negative Mining** | 0.3868 | 0.2636 | 0.7260 | **0.3377** | 0.4214 |
| **6. Selected Matcher + Calibration & Policy** | 0.3868 | **98.34%** | **10.69%** | **1.66%** | 0.4214 |
