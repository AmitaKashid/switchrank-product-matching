import os
import pandas as pd
import numpy as np
from pathlib import Path
from switchrank.models.supervised import SupervisedLightGBMMatcher
from switchrank.evaluation.metrics import compute_all_metrics

PROCESSED_WDC = Path("data/processed/wdc")
REPORTS_DIR = Path("reports")

def main():
    print("=== Running Experiment E3 (Supervised LightGBM Pair Classifier) ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_WDC / "train.csv")
    valid_df = pd.read_csv(PROCESSED_WDC / "valid.csv")
    test_000 = pd.read_csv(PROCESSED_WDC / "test_000un.csv")
    test_050 = pd.read_csv(PROCESSED_WDC / "test_050un.csv")
    test_100 = pd.read_csv(PROCESSED_WDC / "test_100un.csv")

    lgb_model = SupervisedLightGBMMatcher()
    lgb_model.fit(train_df)

    best_t_e3 = lgb_model.fit_threshold(valid_df)
    print(f"E3 LightGBM Optimal Validation Threshold: {best_t_e3:.3f}")

    e3_results = []
    for name, df in [("000un", test_000), ("050un", test_050), ("100un", test_100)]:
        probs = lgb_model.predict_proba(df)
        metrics = compute_all_metrics(df["label"].values, probs, threshold=best_t_e3, hard_neg_mask=df["is_hard_negative"].values)
        print(f"E3 Test {name} -> Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, HardNeg FPR: {metrics['hard_neg_fpr']:.4f}")
        e3_results.append({"experiment": "E3_Supervised_LightGBM", "test_set": name, **metrics})

    pd.DataFrame(e3_results).to_csv(REPORTS_DIR / "supervised_results.csv", index=False)
    print(f"Saved E3 supervised results to {REPORTS_DIR / 'supervised_results.csv'}")

    # Feature Importance Breakdown
    feat_imp = lgb_model.get_feature_importance_df()
    feat_imp.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    print("=== Supervised LightGBM Top Pairwise Feature Importance ===")
    for _, row in feat_imp.iterrows():
        print(f"  {row['feature']:<30}: {row['importance']}")

if __name__ == "__main__":
    main()
