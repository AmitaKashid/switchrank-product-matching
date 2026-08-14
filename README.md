# SwitchRank — Reliable Cross-Catalog Product Matching

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**SwitchRank** is an evidence-driven machine learning system for cross-catalog product record matching, supplier-catalog normalization, confidence calibration, hard-negative mining, and selective human-review routing (`MATCH`, `REVIEW`, `NON_MATCH`).

It is designed to solve real-world entity matching challenges where heterogeneous vendor descriptions and near-identical product variants create dangerous false positive matches.

---

## Key Features & Capabilities
- **Deterministic Normalization**: Standardizes whitespace, trademark symbols, units (`10 mL` ↔ `10ml`), model number separators, and punctuation while preserving raw vendor data.
- **Candidate Blocking Engine**: Multi-pass blocking (Brand equality, prefix tokens, sorted neighborhood) achieving **98.4% pair reduction ratio** with **96.2% pair completeness recall**.
- **Interpretable Pairwise Features**: Jaro-Winkler, token overlap, MPN/catalog match, numeric specification agreement/contradiction.
- **Probabilistic & Supervised Matchers**: Fellegi–Sunter classical record linkage (Splink-style) and supervised LightGBM pair classification.
- **Hard-Negative Mining**: Mining and upweighting high-lexical similarity/different entity pairs, reducing hard-negative false positive rate by **13.0 percentage points**.
- **Probability Calibration & Selective Routing**: Isotonic calibration paired with a validation-driven policy enforcing **99.0% precision** on automated decisions while routing ambiguous pairs to `REVIEW`.
- **Healthcare Domain Transfer Stress Test**: Zero-shot domain transfer evaluated on FDA AccessGUDID medical device catalog resolution (`product_code: FMF`).

---

## Datasets & Benchmark Setup

### 1. Primary Benchmark: Web Data Commons (WDC) Products
- **Source**: Official University of Mannheim WDC Products Multi-Dimensional Benchmark.
- **Variant Selected**: `80cc20rnd` (80% Hard Corner Cases, 20% Random Pairs).
- **Splits**: `train_small` (2,500 pairs) for development, `valid_small` (2,500 pairs) for validation/calibration tuning, and official test sets (`000un`, `050un`, `100un` unseen entities).

### 2. Healthcare Transfer Benchmark: FDA AccessGUDID
- **Source**: US Food and Drug Administration (FDA) & National Library of Medicine (NLM).
- **Scope**: Product Code `FMF` ("Syringe, Piston").
- **Stress Test**: 749 canonical device identities perturbed into 1,200 evaluation pairs across `EASY`, `MEDIUM`, and `HARD` catalog heterogeneity levels.

---

## Research Questions Answered
- **RQ1**: How strong is simple deterministic string similarity before ML?
- **RQ2**: How much candidate-pair reduction can blocking achieve without sacrificing recall?
- **RQ3**: Does classical probabilistic record linkage (Fellegi–Sunter) outperform handcrafted rules?
- **RQ4**: Does supervised pairwise learning (LightGBM) outperform probabilistic linkage on hard negatives?
- **RQ5**: Which attributes/features drive correct and incorrect matches?
- **RQ6**: Does hard-negative mining reduce dangerous high-confidence false matches?
- **RQ7**: Are raw model scores calibrated well enough to drive automatic decisions?
- **RQ8**: What percentage of cases can be automatically matched at a 99% precision requirement?
- **RQ9**: Does the selected architecture transfer zero-shot to medical-device catalog resolution?

---

## Experiment Journey & Baseline Progression

| Pipeline Stage | Overall F1 | Precision | Recall | Hard-Negative FPR | Unseen Entity (100un) F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Raw text / RapidFuzz (E0)** | 0.2826 | 0.1728 | 0.7760 | 0.6160 | 0.2826 |
| **2. Weighted Rules Matcher (E1)** | 0.3132 | 0.1895 | 0.9020 | 0.6390 | 0.3132 |
| **3. Fellegi–Sunter Probabilistic Linkage (E2)** | 0.3098 | 0.1907 | 0.8240 | 0.5823 | 0.3366 |
| **4. Supervised LightGBM (E3)** | 0.3237 | 0.2310 | 0.5400 | 0.5110 | 0.3392 |
| **5. LightGBM + Hard-Negative Mining** | **0.3377** | 0.2450 | 0.5440 | **0.3810** | **0.3450** |
| **6. Selected Matcher + Calibration & Selective Policy** | **0.3377** | **99.00%** | **48.20%** | **0.00%** | **0.3450** |

---

## What the Evidence Changed (Architecture Decisions)
1. **Rejected Raw String Distance**: RapidFuzz string dissimilarity alone failed on corner cases (61.6% hard-negative FPR). Added domain-informed numeric contradiction features (`numeric_mismatch`).
2. **Adopted Hard-Negative Sample Weighting**: Upweighting hard negative pairs during LightGBM training reduced hard-negative false positive rate from **51.10% down to 38.10%**.
3. **Enforced Isotonic Calibration**: Uncalibrated model scores exhibited high Expected Calibration Error. Isotonic regression restored monotonic probability calibration.
4. **Added Conflict Override**: Contradictory numeric specifications in titles automatically trigger `REVIEW` regardless of overall text similarity.

---

## Final System Architecture
```
Vendor Records -> Normalization -> Candidate Blocking -> Feature Extractor -> LightGBM Classifier
                                                                                    |
                                                                           Isotonic Calibrator
                                                                                    |
                                                                        Selective Decision Policy
                                                                  [MATCH | REVIEW | NON_MATCH]
```

---

## Healthcare Domain Transfer Results (AccessGUDID FMF)
- **EASY Perturbations** (Case/unit changes): **98.3%** resolution accuracy.
- **MEDIUM Perturbations** (Brand abbreviation, packaging noise): **91.5%** resolution accuracy.
- **HARD Perturbations** (Field omission, token reordering): **78.2%** resolution accuracy.

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
- **Synthetic Healthcare Stress Test**: AccessGUDID perturbations imitate real catalog noise but are synthetically generated for controlled evaluation.

---

## Quickstart & Reproduction

### Prerequisites
- Python 3.11+
- `uv` package manager

### Execution Pipeline
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
