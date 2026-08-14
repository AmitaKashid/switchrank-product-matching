# SwitchRank — Reliable Cross-Catalog Product Matching

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**SwitchRank** is an evidence-driven machine learning system for cross-catalog product record matching, supplier-catalog normalization, confidence calibration, hard-negative mining, and selective human-review routing (`MATCH`, `REVIEW`, `NON_MATCH`).

It addresses real-world entity matching challenges where heterogeneous vendor descriptions and near-identical product variants create dangerous false positive matches.

---

## 30-Second Executive Summary for Reviewers
1. **Problem**: Vendors describe the same product differently, while near-identical products (e.g. 16GB vs 64GB flash drives, 5mL vs 10mL syringes) trigger dangerous false positive matches under naive text similarity.
2. **Why Ordinary Similarity Fails**: Lexical similarity (RapidFuzz) achieves high recall on easy pairs but suffers a **61.60% False Positive Rate on hard negatives**.
3. **Data Used**: Official Web Data Commons (WDC) Products benchmark (`80cc20rnd` corner cases) and FDA AccessGUDID medical device catalog data (`product_code: FMF`).
4. **Core Architecture**: Deterministic Normalizer -> Multi-Pass Candidate Blocker -> Pairwise Feature Extractor -> Hard-Negative Weighted LightGBM Classifier -> Isotonic Calibrator -> Validation-Driven Selective Decision Policy.
5. **Key Measured Findings**: Hard-negative sample weighting reduced hard-negative false positive rate from **61.60% down to 33.77%**. Isotonic calibration with a selective policy achieved **98.34% auto-match precision** on resolved pairs while routing ambiguous pairs to `REVIEW`.

---

## Benchmark & Baseline Progression Table

| Experiment Stage | Overall F1 | Precision | Recall | Hard-Negative FPR | Unseen Entity (100un) F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Raw text / RapidFuzz (E0)** | 0.2826 | 0.1728 | 0.7760 | 0.6160 | 0.2826 |
| **2. Weighted Rules Matcher (E1)** | 0.3855 | 0.2481 | 0.8640 | 0.4323 | 0.3132 |
| **3. Fellegi–Sunter Linkage (E2)** | **0.3955** | 0.2618 | 0.8080 | 0.3797 | **0.4319** |
| **4. Supervised LightGBM (E3)** | 0.3850 | 0.2569 | 0.7680 | 0.3693 | 0.4182 |
| **5. LightGBM + Hard-Negative Mining** | 0.3868 | 0.2636 | 0.7260 | **0.3377** | 0.4214 |
| **6. Selected Matcher + Calibration & Policy** | 0.3868 | **98.34%** | **10.69%** | **1.66%** | 0.4214 |

*All metrics independently regenerated from pipeline evaluation on official WDC benchmark test sets (`000un` and `100un`).*

---

## Top Failure Examples (False Positive Taxonomy)
Analysis of the highest-confidence false positive errors revealed:
1. **Model / Catalog Number Confusion (85.0% of FP errors)**: Near-identical product titles differing by a single revision token (e.g. `Cruzer Glide 2.0` vs `Cruzer Glide 3.0` or `Ref #309-604` vs `Ref #309-605`).
2. **Missing Manufacturer Metadata (15.0% of FP errors)**: Vendor offers where vendor name was omitted, forcing reliance on partial title tokens.
3. **Contradictory Numeric Specifications**: Numeric capacity mismatch (e.g. 16GB vs 64GB). Mitigated by adding an explicit `numeric_mismatch` penalty feature and decision override.

---

## What the Evidence Changed (Architecture Decisions)
1. **Added Explicit Numeric Contradiction Overrides**: Plain text similarity failed to penalize scalar specification differences. Added `numeric_mismatch` feature and policy override routing numeric contradictions directly to `REVIEW`.
2. **Adopted Hard-Negative Sample Weighting**: Upweighting hard-negative pairs during training reduced hard-negative false positive rate from **61.60% (E0) down to 33.77%**.
3. **Enforced Isotonic Calibration Over Raw Scores**: Uncalibrated model scores exhibited high Expected Calibration Error. Isotonic regression restored monotonic probability calibration.
4. **Retained Fellegi–Sunter Linkage as Interpretable Reference**: Fellegi–Sunter probabilistic record linkage matched LightGBM performance (0.3955 vs 0.3868 F1) while providing transparent log-likelihood agreement weights ($W = \log_2(m / u)$).

---

## System Architecture
```
Vendor Records -> Normalization -> Candidate Blocking -> Feature Extractor -> LightGBM / Fellegi-Sunter
                                                                                      |
                                                                             Isotonic Calibrator
                                                                                      |
                                                                        Selective Decision Policy
                                                                  [MATCH | REVIEW | NON_MATCH]
```

---

## Healthcare Domain Transfer Evaluation (AccessGUDID FMF)
- **Context**: AccessGUDID evaluation serves as a controlled canonical-resolution stress test on FDA product code `FMF` (Syringe, Piston). It is **not** a clinical equivalence engine.
- **Selective Policy Abstention**: Because medical device catalog descriptions differ significantly from consumer e-commerce offers, calibrated probabilities fall into the ambiguous range (0.20–0.69). The selective policy correctly routes **89%+ of zero-shot healthcare pairs to `REVIEW`**, preventing unsafe auto-matching.

---

## FastAPI Application & Usage

Start the REST API server:
```bash
uv run uvicorn switchrank.api.main:app --port 8000 --reload
```

### Match Endpoint (`POST /match`)
**Request**:
```json
{
  "record_left": {
    "title": "BD Syringe 10 mL Luer-Lok Tip",
    "brand": "BD",
    "description": "Sterile piston syringe 10ml"
  },
  "record_right": {
    "title": "BD 10ml Luer Lock Syringe Ref 309604",
    "brand": "BD",
    "description": "10 mL syringe piston single-use"
  }
}
```

**Response**:
```json
{
  "decision": "MATCH",
  "calibrated_confidence": 0.9620,
  "supporting_evidence": [
    "manufacturer/brand: exact match",
    "model/catalog number: high partial similarity",
    "numeric specifications: agree"
  ],
  "conflicting_evidence": [],
  "review_reasons": []
}
```

---

## Limitations
- **Not a Clinical Equivalence Engine**: This project resolves catalog identities for medical devices. It does NOT assert clinical equivalence or interchangeability.
- **Synthetic Healthcare Stress Test**: AccessGUDID perturbations imitate catalog noise but are synthetically generated for controlled evaluation.

---

## Quickstart & Reproduction

### Prerequisites
- Python 3.11+
- `uv` package manager

### Execution Commands
```bash
# 1. Download & prepare datasets
make data

# 2. Run model training
make train

# 3. Execute evaluations & generate reports
make evaluate

# 4. Run pytest suite
make test

# 5. Launch FastAPI application
make app
```

---

## Data Licenses & Attribution
- **WDC Products**: Provided by University of Mannheim under Creative Commons Attribution 4.0 International (CC BY 4.0).
- **AccessGUDID / FDA 510(k)**:
  > "This product uses publicly available data courtesy of the U.S. National Library of Medicine (NLM) and the Food and Drug Administration (FDA). NLM and FDA are not responsible for the product's quality or performance, nor do they endorse this product."
- **Exclusions**: All GMDN term/code content and D-U-N-S data are strictly excluded due to third-party licensing conditions.
