import os
import pandas as pd
import numpy as np
from pathlib import Path
from switchrank.models.supervised import SupervisedLightGBMMatcher
from switchrank.features.extractor import PairFeatureExtractor
from switchrank.calibration.calibrator import ProbabilityCalibrator
from switchrank.decision.policy import SelectiveDecisionPolicy
from switchrank.evaluation.metrics import compute_all_metrics, plot_healthcare_transfer

PROCESSED_WDC = Path("data/processed/wdc")
PROCESSED_GUDID = Path("data/processed/gudid")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("reports/figures")

def main():
    print("=== Running Healthcare Domain Transfer Stress Test (AccessGUDID FMF) ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load frozen model trained on WDC e-commerce products
    train_df = pd.read_csv(PROCESSED_WDC / "train.csv")
    valid_df = pd.read_csv(PROCESSED_WDC / "valid.csv")
    extractor = PairFeatureExtractor()

    weights = np.where(train_df["is_hard_negative"].values == True, 3.0, 1.0)
    model = SupervisedLightGBMMatcher()
    model.fit(train_df, sample_weight=weights)

    val_probs_uncal = model.predict_proba(valid_df)
    iso = ProbabilityCalibrator(method="isotonic")
    iso.fit(val_probs_uncal, valid_df["label"].values)

    policy = SelectiveDecisionPolicy()
    val_feat_df = extractor.transform_df(valid_df)
    policy.fit_thresholds_for_precision(iso.calibrate(val_probs_uncal), valid_df["label"].values, val_feat_df, target_precision=0.99)

    # Load Healthcare Stress Test dataset
    stress_df = pd.read_csv(PROCESSED_GUDID / "fmf_stress_test.csv")
    print(f"Loaded Healthcare Stress Test dataset ({len(stress_df)} pairs).")

    difficulty_metrics = {}
    transfer_records = []

    for diff in ["EASY", "MEDIUM", "HARD"]:
        diff_df = stress_df[stress_df["difficulty"] == diff].copy()
        if len(diff_df) == 0:
            continue

        raw_probs = model.predict_proba(diff_df)
        cal_probs = iso.calibrate(raw_probs)
        feat_df = extractor.transform_df(diff_df)

        metrics = compute_all_metrics(diff_df["label"].values, cal_probs, threshold=policy.match_threshold)
        policy_res = policy.compute_policy_metrics(cal_probs, diff_df["label"].values, feat_df)

        acc = metrics["precision"] * policy_res["auto_match_coverage"] + (1.0 - policy_res["review_rate"]) * 0.1 # top-1 accuracy proxy
        diff_acc = float(np.mean((cal_probs >= policy.match_threshold).astype(int) == diff_df["label"].values))

        difficulty_metrics[diff] = {
            "accuracy": diff_acc,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "coverage": policy_res["auto_match_coverage"],
            "review_rate": policy_res["review_rate"],
        }

        print(f"\n--- Healthcare Transfer Difficulty Level: {diff} ---")
        print(f"  Accuracy           : {diff_acc:.2%}")
        print(f"  F1 Score           : {metrics['f1']:.4f}")
        print(f"  Auto-Match Precision: {policy_res['auto_match_precision']:.2%}")
        print(f"  Auto-Match Coverage : {policy_res['auto_match_coverage']:.2%}")
        print(f"  Human Review Rate   : {policy_res['review_rate']:.2%}")

        transfer_records.append({
            "difficulty": diff,
            "accuracy": round(diff_acc, 4),
            "f1": round(metrics["f1"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "coverage": round(policy_res["auto_match_coverage"], 4),
            "review_rate": round(policy_res["review_rate"], 4),
        })

    # Save results and figure
    pd.DataFrame(transfer_records).to_csv(REPORTS_DIR / "healthcare_transfer_results.csv", index=False)
    plot_healthcare_transfer(difficulty_metrics, FIGURES_DIR / "healthcare_transfer.png")
    print(f"\nSaved healthcare transfer results to {REPORTS_DIR / 'healthcare_transfer_results.csv'} and figure to {FIGURES_DIR / 'healthcare_transfer.png'}")

if __name__ == "__main__":
    main()
