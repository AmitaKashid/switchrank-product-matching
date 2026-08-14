import os
import pandas as pd
import numpy as np
from pathlib import Path
from switchrank.models.supervised import SupervisedLightGBMMatcher
from switchrank.features.extractor import PairFeatureExtractor
from switchrank.calibration.calibrator import ProbabilityCalibrator, compute_calibration_metrics
from switchrank.decision.policy import SelectiveDecisionPolicy
from switchrank.evaluation.metrics import plot_reliability_diagram, plot_precision_coverage

PROCESSED_WDC = Path("data/processed/wdc")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("reports/figures")

def main():
    print("=== Running Calibration & Selective Decision Policy Evaluation ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_WDC / "train.csv")
    valid_df = pd.read_csv(PROCESSED_WDC / "valid.csv")
    test_000 = pd.read_csv(PROCESSED_WDC / "test_000un.csv")

    extractor = PairFeatureExtractor()

    # Train base LightGBM model
    weights = np.where(train_df["is_hard_negative"].values == True, 3.0, 1.0)
    model = SupervisedLightGBMMatcher()
    model.fit(train_df, sample_weight=weights)

    val_probs_uncal = model.predict_proba(valid_df)
    test_probs_uncal = model.predict_proba(test_000)
    y_val = valid_df["label"].values
    y_test = test_000["label"].values

    # Fit Platt and Isotonic Calibrators
    platt = ProbabilityCalibrator(method="platt")
    platt.fit(val_probs_uncal, y_val)
    test_probs_platt = platt.calibrate(test_probs_uncal)

    iso = ProbabilityCalibrator(method="isotonic")
    iso.fit(val_probs_uncal, y_val)
    test_probs_iso = iso.calibrate(test_probs_uncal)

    # Compute Calibration Metrics
    uncal_metrics = compute_calibration_metrics(test_probs_uncal, y_test)
    platt_metrics = compute_calibration_metrics(test_probs_platt, y_test)
    iso_metrics = compute_calibration_metrics(test_probs_iso, y_test)

    print("\n=== Calibration Performance Comparison ===")
    print(f"Uncalibrated -> ECE: {uncal_metrics['expected_calibration_error']:.4f}, Brier: {uncal_metrics['brier_score']:.4f}, LogLoss: {uncal_metrics['log_loss']:.4f}")
    print(f"Platt Sigmoid -> ECE: {platt_metrics['expected_calibration_error']:.4f}, Brier: {platt_metrics['brier_score']:.4f}, LogLoss: {platt_metrics['log_loss']:.4f}")
    print(f"Isotonic     -> ECE: {iso_metrics['expected_calibration_error']:.4f}, Brier: {iso_metrics['brier_score']:.4f}, LogLoss: {iso_metrics['log_loss']:.4f}")

    # Plot Reliability Curve
    plot_reliability_diagram(y_test, test_probs_uncal, test_probs_iso, FIGURES_DIR / "reliability_diagram.png")
    print(f"Saved reliability diagram to {FIGURES_DIR / 'reliability_diagram.png'}")

    # Evaluate Selective Decision Policy
    val_feat_df = extractor.transform_df(valid_df)
    test_feat_df = extractor.transform_df(test_000)

    val_probs_cal = iso.calibrate(val_probs_uncal)

    policy = SelectiveDecisionPolicy()
    match_t, non_t = policy.fit_thresholds_for_precision(val_probs_cal, y_val, val_feat_df, target_precision=0.99)

    policy_metrics = policy.compute_policy_metrics(test_probs_iso, y_test, test_feat_df)
    print("\n=== Selective Decision Policy Test Results (Target 99% Precision) ===")
    print(f"  Auto-Match Precision : {policy_metrics['auto_match_precision']:.2%}")
    print(f"  Auto-Match Coverage  : {policy_metrics['auto_match_coverage']:.2%}")
    print(f"  Human Review Rate    : {policy_metrics['review_rate']:.2%}")
    print(f"  False Auto-Match Rate: {policy_metrics['false_auto_match_rate']:.2%}")

    # Generate Precision-Coverage Frontier Data
    coverages = []
    precisions = []
    for t_target in np.linspace(0.85, 0.99, 15):
        pol_temp = SelectiveDecisionPolicy()
        pol_temp.fit_thresholds_for_precision(val_probs_cal, y_val, val_feat_df, target_precision=t_target)
        m_temp = pol_temp.compute_policy_metrics(test_probs_iso, y_test, test_feat_df)
        coverages.append(m_temp["auto_match_coverage"])
        precisions.append(m_temp["auto_match_precision"])

    plot_precision_coverage(coverages, precisions, FIGURES_DIR / "precision_coverage_curve.png")
    print(f"Saved precision-coverage curve to {FIGURES_DIR / 'precision_coverage_curve.png'}")

if __name__ == "__main__":
    main()
