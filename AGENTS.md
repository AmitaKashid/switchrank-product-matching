# AGENTS.md — Permanent Project Contract

## 1. Project Overview & Identity
- **Project Name**: SwitchRank — Reliable Cross-Catalog Product Matching
- **Purpose**: Demonstrate decision-oriented machine learning for entity/product matching, supplier-catalog normalization, confidence calibration, hard-negative mining, and selective human-review routing (`MATCH`, `REVIEW`, `NON_MATCH`).
- **Target Repository**: `switchrank-product-matching` (authenticated GitHub account).

## 2. Core Operational Constraints & Exclusions
- **DO NOT build around**: RAG, BM25, BGE-M3, vector databases, cross-encoder retrieval engines, LangGraph, MCP, agent orchestration, generic semantic search, chatbots.
- **Problem Focus**: Pairwise entity/product matching and reliable decision-making under domain transfer and hard negatives.
- **Truthfulness**: Never fabricate metrics, figures, or experiment results. Every result recorded in `docs/DECISIONS.md` or `reports/` must stem from actual pipeline execution.
- **Medical Disclaimer**: This project is NOT a medical clinical-equivalence engine. The healthcare component models canonical medical-device catalog record resolution. Never claim two different medical devices are clinically interchangeable.

## 3. Dataset Specifications & Licensing Rules
- **Primary Benchmark**: Web Data Commons (WDC) Products Multi-Dimensional Benchmark (`80cc20rnd` corner-case variant, using official Mannheim downloads, evaluated on 0%, 50%, and 100% unseen entity test sets).
- **Healthcare Transfer Benchmark**: NLM AccessGUDID / FDA Product Code `FMF` (Syringe, Piston).
- **Licensing Restrictions**: Exclude GMDN term/code/definition content and D-U-N-S data. Raw third-party data must not be committed to Git unless explicitly permitted. Use reproducible download scripts under `scripts/`.

## 4. Required Research Questions (RQs)
- **RQ1**: Baseline string/token similarity performance (E0).
- **RQ2**: Blocking strategy candidate pair reduction vs. true-match recall.
- **RQ3**: Classical Fellegi–Sunter probabilistic record linkage vs. handcrafted rules (E1 vs. E2).
- **RQ4**: Supervised LightGBM pair classifier performance on hard negatives and unseen entities (E3).
- **RQ5**: SHAP/Feature importance analysis of matching decisions.
- **RQ6**: Hard-negative mining impact on false-positive rates.
- **RQ7**: Score calibration quality (Brier score, ECE, reliability diagrams).
- **RQ8**: Selective decision policy precision-coverage frontier (`MATCH` / `REVIEW` / `NON_MATCH`).
- **RQ9**: Domain transfer generalization to medical-device catalog resolution (AccessGUDID stress test).

## 5. Definition of Done
- Complete data acquisition scripts (`scripts/download_wdc.py`, `scripts/download_gudid.py`, `scripts/prepare_data.py`).
- Deterministic normalization, multi-pass blocking, interpretable features, probabilistic linkage, LightGBM classifier, calibration, selective decision policy, FastAPI application (`POST /match`, `GET /health`).
- Pytest suite (`tests/`) covering normalization, blocking, features, leakage prevention, policy, and API endpoints.
- Results artifacts (`reports/results.csv`, `reports/ablation.csv`, `reports/error_analysis.csv`, `reports/final_report.md`, and figures under `reports/figures/`).
- Documentation (`AGENTS.md`, `README.md`, `LICENSE`, `docs/PROJECT_PLAN.md`, `docs/DATA_AUDIT.md`, `docs/DATA_LICENSES.md`, `docs/DECISIONS.md`).
- Git history and GitHub repository publication.
