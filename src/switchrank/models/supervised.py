import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List
from switchrank.features.extractor import PairFeatureExtractor, FEATURE_NAMES

class SupervisedLightGBMMatcher:
    """Experiment E3: Supervised LightGBM pair classification engine."""

    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.threshold = 0.5
        self.extractor = PairFeatureExtractor()
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=31,
            random_state=self.random_state,
            verbosity=-1,
        )

    def fit(self, train_df: pd.DataFrame, sample_weight: np.ndarray = None):
        X_train = self.extractor.transform_df(train_df)
        y_train = train_df["label"].values

        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)

    def fit_threshold(self, val_df: pd.DataFrame) -> float:
        X_val = self.extractor.transform_df(val_df)
        y_val = val_df["label"].values
        probs = self.model.predict_proba(X_val)[:, 1]

        best_f1 = -1.0
        best_t = 0.5
        for t in np.linspace(0.1, 0.9, 81):
            preds = (probs >= t).astype(int)
            tp = np.sum((preds == 1) & (y_val == 1))
            fp = np.sum((preds == 1) & (y_val == 0))
            fn = np.sum((preds == 0) & (y_val == 1))

            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = (2 * prec * rec) / max(prec + rec, 1e-6)

            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        self.threshold = best_t
        return best_t

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.extractor.transform_df(df)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(df)
        return (probs >= self.threshold).astype(int)

    def get_feature_importance_df(self) -> pd.DataFrame:
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": FEATURE_NAMES,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
