import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz
from typing import Dict, Any, List

class RapidFuzzBaseline:
    """Experiment E0: Simple string dissimilarity baseline using RapidFuzz."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def predict_pair(self, title_left: str, title_right: str) -> float:
        if not title_left or not title_right or pd.isna(title_left) or pd.isna(title_right):
            return 0.0
        score = fuzz.token_set_ratio(str(title_left), str(title_right)) / 100.0
        return float(score)

    def fit_threshold(self, val_df: pd.DataFrame) -> float:
        """Find optimal threshold on validation data to maximize F1."""
        best_f1 = -1.0
        best_t = 0.5
        scores = [self.predict_pair(r.title_left, r.title_right) for _, r in val_df.iterrows()]
        y_true = val_df["label"].values

        for t in np.linspace(0.3, 0.95, 66):
            preds = (np.array(scores) >= t).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))

            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = (2 * prec * rec) / max(prec + rec, 1e-6)

            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        self.threshold = best_t
        return best_t

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_pair(r.title_left, r.title_right) for _, r in df.iterrows()])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(df)
        return (probs >= self.threshold).astype(int)
