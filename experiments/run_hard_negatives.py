import os
import pandas as pd
import numpy as np
from pathlib import Path
from switchrank.models.supervised import SupervisedLightGBMMatcher
from switchrank.features.extractor import PairFeatureExtractor
from switchrank.evaluation.metrics import compute_all_metrics

PROCESSED_WDC = Path("data/processed/wdc")
REPORTS_DIR = Path("reports")

def main():
    print("=== Running Hard-Negative Mining & Error Taxonomy Analysis ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_WDC / "train.csv")
    valid_df = pd.read_csv(PROCESSED_WDC / "valid.csv")
    test_000 = pd.read_csv(PROCESSED_WDC / "test_000un.csv")

    # Standard model
    model_std = SupervisedLightGBMMatcher()
    model_std.fit(train_df)
    t_std = model_std.fit_threshold(valid_df)
    probs_std = model_std.predict_proba(test_000)
    metrics_std = compute_all_metrics(test_000["label"].values, probs_std, threshold=t_std, hard_neg_mask=test_000["is_hard_negative"].values)

    # Hard negative upweighted model
    weights = np.where(train_df["is_hard_negative"].values == True, 3.0, 1.0)
    model_hn = SupervisedLightGBMMatcher()
    model_hn.fit(train_df, sample_weight=weights)
    t_hn = model_hn.fit_threshold(valid_df)
    probs_hn = model_hn.predict_proba(test_000)
    metrics_hn = compute_all_metrics(test_000["label"].values, probs_hn, threshold=t_hn, hard_neg_mask=test_000["is_hard_negative"].values)

    print(f"Standard LightGBM -> Hard-Neg FPR: {metrics_std['hard_neg_fpr']:.4f}, Overall F1: {metrics_std['f1']:.4f}")
    print(f"Hard-Neg Weighted -> Hard-Neg FPR: {metrics_hn['hard_neg_fpr']:.4f}, Overall F1: {metrics_hn['f1']:.4f}")

    # Top False Positives Error Taxonomy Analysis
    test_000["std_prob"] = probs_std
    test_000["pred"] = (probs_std >= t_std).astype(int)

    fps = test_000[(test_000["label"] == 0) & (test_000["pred"] == 1)].sort_values(by="std_prob", ascending=False)
    print(f"\nAnalyzing Top {min(20, len(fps))} False Positive Errors:")

    taxonomy_counts = {
        "Capacity/Size Mismatch": 0,
        "Model/Catalog Number Confusion": 0,
        "Accessory vs Primary Product": 0,
        "Missing Manufacturer/Brand": 0,
        "Ambiguous Short Title": 0,
    }

    error_records = []
    for idx, row in fps.head(20).iterrows():
        t1, t2 = str(row["title_left"]), str(row["title_right"])
        p = row["std_prob"]

        # Rule-based error categorization
        if any(c in t1.lower() or c in t2.lower() for c in ["gb", "tb", "ml", "mm"]) and ("16" in t1 or "64" in t1 or "128" in t1 or "256" in t1):
            category = "Capacity/Size Mismatch"
        elif any(char.isdigit() for char in t1) and any(char.isdigit() for char in t2):
            category = "Model/Catalog Number Confusion"
        elif any(a in t1.lower() or a in t2.lower() for a in ["case", "cover", "cable", "adapter", "refill"]):
            category = "Accessory vs Primary Product"
        elif pd.isna(row.get("brand_left")) or pd.isna(row.get("brand_right")):
            category = "Missing Manufacturer/Brand"
        else:
            category = "Ambiguous Short Title"

        taxonomy_counts[category] += 1
        error_records.append({
            "pair_id": row["pair_id"],
            "title_left": t1[:60],
            "title_right": t2[:60],
            "predicted_prob": round(p, 4),
            "error_category": category,
        })

    print("=== False Positive Error Taxonomy Distribution ===")
    for cat, count in taxonomy_counts.items():
        print(f"  {cat:<35}: {count} cases ({count/max(len(error_records), 1):.1%})")

    err_df = pd.DataFrame(error_records)
    err_df.to_csv(REPORTS_DIR / "error_analysis.csv", index=False)
    print(f"Saved error taxonomy breakdown to {REPORTS_DIR / 'error_analysis.csv'}")

if __name__ == "__main__":
    main()
