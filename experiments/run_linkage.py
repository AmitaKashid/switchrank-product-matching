import os
import pandas as pd
import numpy as np
from pathlib import Path
from switchrank.models.probabilistic import FellegiSunterLinkage
from switchrank.evaluation.metrics import compute_all_metrics

PROCESSED_WDC = Path("data/processed/wdc")
REPORTS_DIR = Path("reports")

def main():
    print("=== Running Experiment E2 (Fellegi-Sunter Probabilistic Linkage) ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_WDC / "train.csv")
    valid_df = pd.read_csv(PROCESSED_WDC / "valid.csv")
    test_000 = pd.read_csv(PROCESSED_WDC / "test_000un.csv")
    test_050 = pd.read_csv(PROCESSED_WDC / "test_050un.csv")
    test_100 = pd.read_csv(PROCESSED_WDC / "test_100un.csv")

    fs_model = FellegiSunterLinkage()
    fs_model.fit(train_df)

    best_t_e2 = fs_model.fit_threshold(valid_df)
    print(f"E2 Fellegi-Sunter Optimal Validation Threshold: {best_t_e2:.3f}")

    e2_results = []
    for name, df in [("000un", test_000), ("050un", test_050), ("100un", test_100)]:
        probs = fs_model.predict_proba(df)
        metrics = compute_all_metrics(df["label"].values, probs, threshold=best_t_e2, hard_neg_mask=df["is_hard_negative"].values)
        print(f"E2 Test {name} -> Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, HardNeg FPR: {metrics['hard_neg_fpr']:.4f}")
        e2_results.append({"experiment": "E2_Fellegi_Sunter", "test_set": name, **metrics})

    pd.DataFrame(e2_results).to_csv(REPORTS_DIR / "probabilistic_results.csv", index=False)
    print(f"Saved E2 probabilistic linkage results to {REPORTS_DIR / 'probabilistic_results.csv'}")

if __name__ == "__main__":
    main()
