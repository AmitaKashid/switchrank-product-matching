# PROJECT PLAN — SwitchRank: Reliable Cross-Catalog Product Matching

## 1. Technical Context & Scope
Product entity matching is a critical challenge in e-commerce and supply chain intelligence: distinct vendors describe identical physical products with varying titles, abbreviations, model numbers, and formatting. Misclassifying near-identical products (e.g. 16GB vs 64GB flash drives, 5mL vs 10mL syringes) creates high-risk false matches.

SwitchRank implements an evidence-driven, multi-stage machine learning system that pairs candidate products, computes interpretable similarity metrics, models matching evidence probabilistic and supervised classifiers, calibrates decision probabilities, and enforces a selective decision policy (`MATCH`, `REVIEW`, `NON_MATCH`).

## 2. Research Questions
1. **RQ1**: How strong is simple deterministic normalization/string similarity before introducing ML?
2. **RQ2**: How much candidate-pair reduction can blocking achieve without sacrificing true-match recall?
3. **RQ3**: Does classical probabilistic record linkage (Fellegi–Sunter) outperform handcrafted matching rules?
4. **RQ4**: Does supervised pairwise learning (LightGBM) materially outperform probabilistic linkage, particularly on hard negatives and unseen entities?
5. **RQ5**: Which attributes/features actually drive correct and incorrect matches?
6. **RQ6**: Does hard-negative mining reduce the dangerous failure mode of assigning high confidence to visually similar but different products?
7. **RQ7**: Are raw model scores calibrated well enough to drive automatic decisions?
8. **RQ8**: What percentage of cases can be automatically matched at a high precision requirement (target 99%), while sending ambiguous cases to REVIEW?
9. **RQ9**: Does the selected architecture transfer to medical-device catalog resolution, or does its performance collapse under a new domain?

## 3. Methodological Workflow & Architecture
```
Raw Records -> Normalization -> Candidate Blocking -> Feature Extraction
                                                            |
                                    +-----------------------+-----------------------+
                                    |                       |                       |
                                Rule Matcher         Fellegi-Sunter             LightGBM
                                 (Baseline)            (Probabilistic)         (Supervised)
                                    |                       |                       |
                                    +-----------------------+-----------------------+
                                                            |
                                                   Platt/Isotonic Calibration
                                                            |
                                                Selective Decision Policy
                                           [MATCH | REVIEW | NON_MATCH]
```

## 4. Evaluation Strategy & Metrics
- **Performance Metrics**: Precision, Recall, F1-Score, PR-AUC, Confusion Matrix.
- **Reliability Metrics**: Expected Calibration Error (ECE), Brier Score, Reliability Curves.
- **Decision Metrics**: Auto-match Precision, Coverage %, Review Rate %, False Auto-match Rate %.
- **Domain Transfer**: Top-1 resolution accuracy across EASY, MEDIUM, and HARD catalog perturbations on AccessGUDID FMF medical devices.
