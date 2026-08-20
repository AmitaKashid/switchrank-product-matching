# FINAL AUDIT & REPAIR LOG — SwitchRank


---

## 1. Executive Summary & Audit Scope
This audit independently inspects, challenges, repairs, and validates the entire `switchrank-product-matching` repository. Every claim, baseline, metric, feature, calibration step, and domain transfer test was audited against code execution. All public documentation reads from `reports/final_metrics.json`.

---

## 2. Audit Findings & Reconciled Resolutions

| ID | Category | Discovered Issue | Severity | Status / Resolution |
| :--- | :--- | :--- | :---: | :--- |
| **AUD-01** | **Metric Reconciliation & Source of Truth** | Previous draft reports contained conflicting metrics across legacy experiment scripts. | **HIGH** | **RECONCILED**: Established single canonical pipeline (`scripts/generate_reports.py`) generating authoritative `reports/final_metrics.json`. All public markdown files read from `final_metrics.json`. |
| **AUD-02** | **Healthcare Domain-Shift Wording** | Previous text ambiguously described medical device evaluation as "resolution accuracy". | **HIGH** | **RECONCILED**: Clarified dataset as **FDA 510(k) Medical Device Clearances Database (Product Code FMF)** and framed as a zero-shot domain-shift/abstention experiment where the selective policy abstained on 99.92% of cases (1,199 / 1,200 routed to REVIEW). |
| **AUD-03** | **Causal Hard-Negative Weighting Claim** | Previous text attributed the entire FPR reduction from RapidFuzz (59.73%) to sample weighting. | **HIGH** | **RECONCILED**: Direct causal intervention comparison measures Standard LightGBM (39.33% FPR) vs Hard-Negative Weighted LightGBM (33.77% FPR), yielding **-5.57 percentage points (-14.15% relative)** FPR reduction. |
| **AUD-04** | **Candidate Blocking Limitation** | 98.13% pair reduction was previously presented without qualification. | **MEDIUM** | **RECONCILED**: Explicitly documented that multi-pass blocking retains **70.80% blocking recall**, which is a known limitation requiring higher-recall candidate generation in production. |
| **AUD-05** | **Calibration Split Leakage Safety** | Validation data required strict split into fitting and policy evaluation subsets. | **HIGH** | **RECONCILED**: Split `valid_small` (2,500 pairs) into `val_calib` (1,250 pairs) for fitting calibrators and `val_policy` (1,250 pairs) for evaluating ECE/Brier/LogLoss and policy thresholds. Isotonic regression selected (ECE = 0.0290 vs uncalibrated 0.0798). |

---

## 3. Reconciled Baseline & Ablation Table (`reports/final_metrics.json`)

| Pipeline Stage | 000un F1 | 050un F1 | 100un F1 | Precision | Recall | Hard-Negative FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Raw text / RapidFuzz (E0)** | 0.2804 | 0.3080 | 0.2982 | 0.1724 | 0.7500 | 0.5973 |
| **2. Weighted Rules Matcher (E1)** | 0.3881 | 0.4148 | 0.4205 | 0.2504 | 0.8620 | 0.4267 |
| **3. Fellegi–Sunter Linkage (E2)** | **0.3955** | **0.4252** | **0.4319** | 0.2618 | 0.8080 | 0.3797 |
| **4. Supervised LightGBM (E3)** | 0.3817 | 0.4105 | 0.4110 | 0.2513 | 0.7940 | 0.3933 |
| **5. LightGBM + Hard-Negative Mining** | 0.3868 | 0.4208 | 0.4214 | 0.2636 | 0.7260 | **0.3377** |
| **6. Selected Matcher + Calibration & Policy** | 0.3868 | 0.4208 | 0.4214 | **98.34%** | **10.71%** | **1.66%** |
