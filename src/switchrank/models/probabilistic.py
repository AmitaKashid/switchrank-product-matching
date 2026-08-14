import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from switchrank.features.extractor import PairFeatureExtractor

class FellegiSunterLinkage:
    """Experiment E2: Fellegi-Sunter Classical Probabilistic Record Linkage Engine."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.extractor = PairFeatureExtractor()
        self.m_probs: Dict[str, float] = {}
        self.u_probs: Dict[str, float] = {}
        self.log_weights: Dict[str, float] = {}
        self.prior_m: float = 0.5

    def fit(self, train_df: pd.DataFrame):
        """Fit m-probabilities (P(agree|Match)) and u-probabilities (P(agree|NonMatch))."""
        feat_df = self.extractor.transform_df(train_df)
        labels = train_df["label"].values

        matches = feat_df[labels == 1]
        non_matches = feat_df[labels == 0]

        self.prior_m = float(np.mean(labels))
        eps = 1e-4

        for col in feat_df.columns:
            # Binary agreement indicator thresholded at 0.5
            m_agree = np.mean(matches[col] >= 0.5)
            u_agree = np.mean(non_matches[col] >= 0.5)

            # Laplace smoothing
            m_agree = float(np.clip(m_agree, eps, 1.0 - eps))
            u_agree = float(np.clip(u_agree, eps, 1.0 - eps))

            self.m_probs[col] = m_agree
            self.u_probs[col] = u_agree

            # Fellegi-Sunter weight: log2(m / u) for agreement, log2((1-m) / (1-u)) for disagreement
            w_agree = np.log2(m_agree / u_agree)
            self.log_weights[col] = float(w_agree)

        print("=== Fellegi-Sunter Field Agreement Weights ===")
        for k, v in self.log_weights.items():
            print(f"  Field '{k}': m={self.m_probs[k]:.3f}, u={self.u_probs[k]:.3f} => Weight={v:+.3f}")

    def predict_pair_prob(self, feat: Dict[str, float]) -> float:
        w_total = np.log2(self.prior_m / max(1.0 - self.prior_m, 1e-6))
        for col, w_agree in self.log_weights.items():
            val = feat.get(col, 0.0)
            m = self.m_probs[col]
            u = self.u_probs[col]
            if val >= 0.5:
                w_total += np.log2(m / u)
            else:
                w_total += np.log2((1.0 - m) / (1.0 - u))

        # Posterior probability P(Match | gamma)
        odds = 2.0 ** w_total
        prob = odds / (1.0 + odds)
        return float(prob)

    def fit_threshold(self, val_df: pd.DataFrame) -> float:
        feat_df = self.extractor.transform_df(val_df)
        probs = np.array([self.predict_pair_prob(row.to_dict()) for _, row in feat_df.iterrows()])
        y_true = val_df["label"].values

        best_f1 = -1.0
        best_t = 0.5
        for t in np.linspace(0.1, 0.9, 81):
            preds = (probs >= t).astype(int)
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
        return np.array([self.predict_pair_prob(row.to_dict()) for _, row in feat_df.iterrows()])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(df)
        return (probs >= self.threshold).astype(int)
