import os
import pandas as pd
import numpy as np
from pathlib import Path
from switchrank.models.baseline import RapidFuzzBaseline
from switchrank.models.rule_matcher import WeightedRuleMatcher
from switchrank.evaluation.metrics import compute_all_metrics

PROCESSED_WDC = Path("data/processed/wdc")
REPORTS_DIR = Path("reports")

def main():
    print("=== Running Experiment E0 (RapidFuzz) & E1 (Weighted Rules) ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_WDC / "train.csv")
    valid_df = pd.read_csv(PROCESSED_WDC / "valid.csv")
    test_000 = pd.read_csv(PROCESSED_WDC / "test_000un.csv")
    test_050 = pd.read_csv(PROCESSED_WDC / "test_050un.csv")
    test_100 = pd.read_csv(PROCESSED_WDC / "test_100un.csv")

    # E0 RapidFuzz
    e0_model = RapidFuzzBaseline()
    best_t_e0 = e0_model.fit_threshold(valid_df)
    print(f"E0 RapidFuzz Optimal Validation Threshold: {best_t_e0:.3f}")

    e0_probs = e0_model.predict_proba(test_000)
    e0_metrics = compute_all_metrics(test_000["label"].values, e0_probs, threshold=best_t_e0, hard_neg_mask=test_000["is_hard_negative"].values)
    print(f"E0 Test 000un -> Prec: {e0_metrics['precision']:.4f}, Rec: {e0_metrics['recall']:.4f}, F1: {e0_metrics['f1']:.4f}, PR-AUC: {e0_metrics['pr_auc']:.4f}, HardNeg FPR: {e0_metrics['hard_neg_fpr']:.4f}")

    # E1 Weighted Rules
    e1_model = WeightedRuleMatcher()
    best_t_e1 = e1_model.fit_threshold(valid_df)
    print(f"E1 Weighted Rules Optimal Validation Threshold: {best_t_e1:.3f}")

    e1_probs = e1_model.predict_proba(test_000)
    e1_metrics = compute_all_metrics(test_000["label"].values, e1_probs, threshold=best_t_e1, hard_neg_mask=test_000["is_hard_negative"].values)
    print(f"E1 Test 000un -> Prec: {e1_metrics['precision']:.4f}, Rec: {e1_metrics['recall']:.4f}, F1: {e1_metrics['f1']:.4f}, PR-AUC: {e1_metrics['pr_auc']:.4f}, HardNeg FPR: {e1_metrics['hard_neg_fpr']:.4f}")

    # Save baseline metrics
    baseline_records = [
        {"experiment": "E0_RapidFuzz_Baseline", "test_set": "000un", **e0_metrics},
        {"experiment": "E1_Weighted_Rules", "test_set": "000un", **e1_metrics},
    ]
    pd.DataFrame(baseline_records).to_csv(REPORTS_DIR / "baselines_results.csv", index=False)
    print(f"Saved baseline evaluation results to {REPORTS_DIR / 'baselines_results.csv'}")

if __name__ == "__main__":
    main()
