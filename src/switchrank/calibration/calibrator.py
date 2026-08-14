import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

class ProbabilityCalibrator:
    """Platt Sigmoid and Isotonic probability calibrator for pair matching scores."""

    def __init__(self, method: str = "platt"):
        self.method = method
        self.is_fitted = False
        if method == "platt":
            self.model = LogisticRegression(C=1.0, solver="lbfgs")
        elif method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def fit(self, uncalibrated_scores: np.ndarray, y_true: np.ndarray):
        scores_2d = uncalibrated_scores.reshape(-1, 1)
        if self.method == "platt":
            self.model.fit(scores_2d, y_true)
        else:
            self.model.fit(uncalibrated_scores, y_true)
        self.is_fitted = True

    def calibrate(self, uncalibrated_scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return uncalibrated_scores
        if self.method == "platt":
            return self.model.predict_proba(uncalibrated_scores.reshape(-1, 1))[:, 1]
        else:
            return self.model.predict(uncalibrated_scores)

def compute_calibration_metrics(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Compute Brier Score, Log Loss, and Expected Calibration Error (ECE)."""
    brier = float(brier_score_loss(y_true, probs))
    loss = float(log_loss(y_true, np.clip(probs, 1e-6, 1.0 - 1e-6)))

    # Compute ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return {
        "brier_score": brier,
        "log_loss": loss,
        "expected_calibration_error": float(ece),
    }
