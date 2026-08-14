import numpy as np
import pandas as pd
from typing import Dict, Any, List
from switchrank.features.extractor import PairFeatureExtractor

class WeightedRuleMatcher:
    """Experiment E1: Interpretable weighted compatibility score rule matcher."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.extractor = PairFeatureExtractor()
        # Initial domain-informed heuristic weights
        self.weights = {
            "title_token_set_ratio": 0.30,
            "title_token_overlap": 0.25,
            "brand_exact": 0.15,
            "model_exact_match": 0.25,
            "numeric_token_match": 0.15,
            "numeric_mismatch": -0.35, # strong penalty for numeric contradiction
        }

    def predict_pair_score(self, feat: Dict[str, float]) -> float:
        score = 0.0
        for k, w in self.weights.items():
            score += feat.get(k, 0.0) * w
        # Sigmoid squash to [0, 1] range
        prob = 1.0 / (1.0 + np.exp(-5.0 * (score - 0.3)))
        return float(prob)

    def fit_threshold(self, val_df: pd.DataFrame) -> float:
        feat_df = self.extractor.transform_df(val_df)
        scores = [self.predict_pair_score(row.to_dict()) for _, row in feat_df.iterrows()]
        y_true = val_df["label"].values

        best_f1 = -1.0
        best_t = 0.5
        for t in np.linspace(0.2, 0.9, 71):
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
        feat_df = self.extractor.transform_df(df)
        return np.array([self.predict_pair_score(row.to_dict()) for _, row in feat_df.iterrows()])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(df)
        return (probs >= self.threshold).astype(int)
