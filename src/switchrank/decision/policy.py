import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

DECISION_MATCH = "MATCH"
DECISION_REVIEW = "REVIEW"
DECISION_NON_MATCH = "NON_MATCH"

class SelectiveDecisionPolicy:
    """Validation-driven selective decision policy for automatic matching and human-review routing."""

    def __init__(self, match_threshold: float = 0.85, non_match_threshold: float = 0.20):
        self.match_threshold = match_threshold
        self.non_match_threshold = non_match_threshold

    def fit_thresholds_for_precision(
        self, calibrated_probs: np.ndarray, y_true: np.ndarray, feature_df: pd.DataFrame, target_precision: float = 0.99
    ) -> Tuple[float, float]:
        """Find match and non-match thresholds on validation set to achieve target precision."""
        best_match_t = 0.85
        best_non_match_t = 0.20
        best_coverage = 0.0

        for match_t in np.linspace(0.60, 0.98, 39):
            for non_t in np.linspace(0.05, 0.40, 36):
                if non_t >= match_t:
                    continue

                # Make decisions
                decisions = []
                for p, (_, feat) in zip(calibrated_probs, feature_df.iterrows()):
                    # Conflict override: numeric mismatch triggers REVIEW regardless of score
                    if feat.get("numeric_mismatch", 0.0) == 1.0:
                        decisions.append(DECISION_REVIEW)
                    elif p >= match_t:
                        decisions.append(DECISION_MATCH)
                    elif p <= non_t:
                        decisions.append(DECISION_NON_MATCH)
                    else:
                        decisions.append(DECISION_REVIEW)

                decisions = np.array(decisions)
                auto_mask = decisions != DECISION_REVIEW
                coverage = np.mean(auto_mask)

                if np.sum(auto_mask) == 0:
                    continue

                # Evaluate auto-decision correctness
                auto_correct = 0
                for d, label in zip(decisions[auto_mask], y_true[auto_mask]):
                    if d == DECISION_MATCH and label == 1:
                        auto_correct += 1
                    elif d == DECISION_NON_MATCH and label == 0:
                        auto_correct += 1

                precision = auto_correct / np.sum(auto_mask)

                if precision >= target_precision and coverage > best_coverage:
                    best_coverage = coverage
                    best_match_t = float(match_t)
                    best_non_match_t = float(non_t)

        self.match_threshold = best_match_t
        self.non_match_threshold = best_non_match_t
        print(f"Selected Policy Thresholds for Target Precision {target_precision:.2%}: Match>={self.match_threshold:.3f}, NonMatch<={self.non_match_threshold:.3f} (Validation Coverage: {best_coverage:.2%})")
        return best_match_t, best_non_match_t

    def evaluate_record_pair(self, prob: float, feat: Dict[str, float]) -> Dict[str, Any]:
        supporting = []
        conflicting = []
        review_reasons = []

        # Analyze evidence
        if feat.get("brand_exact", 0.0) == 1.0:
            supporting.append("manufacturer/brand: exact match")
        elif feat.get("brand_missing", 0.0) == 1.0:
            conflicting.append("manufacturer/brand: missing on one or both records")

        if feat.get("model_exact_match", 0.0) == 1.0:
            supporting.append("model/catalog number: exact agreement")
        elif feat.get("model_partial_match", 0.0) >= 0.7:
            supporting.append("model/catalog number: high partial similarity")

        if feat.get("numeric_token_match", 0.0) == 1.0:
            supporting.append("numeric specifications: agree")
        elif feat.get("numeric_mismatch", 0.0) == 1.0:
            conflicting.append("numeric specifications: mismatch detected")
            review_reasons.append("Contradictory numeric specs in titles")

        if feat.get("title_token_overlap", 0.0) >= 0.6:
            supporting.append("title tokens: high overlap")

        # Determine decision
        if feat.get("numeric_mismatch", 0.0) == 1.0:
            decision = DECISION_REVIEW
            if not review_reasons:
                review_reasons.append("Numeric spec conflict")
        elif prob >= self.match_threshold:
            decision = DECISION_MATCH
        elif prob <= self.non_match_threshold:
            decision = DECISION_NON_MATCH
        else:
            decision = DECISION_REVIEW
            review_reasons.append(f"Calibrated probability ({prob:.3f}) falls in ambiguous range ({self.non_match_threshold:.2f} - {self.match_threshold:.2f})")

        return {
            "decision": decision,
            "calibrated_confidence": float(prob),
            "supporting_evidence": supporting,
            "conflicting_evidence": conflicting,
            "review_reasons": review_reasons,
        }

    def predict_policy(self, calibrated_probs: np.ndarray, feature_df: pd.DataFrame) -> List[Dict[str, Any]]:
        results = []
        for p, (_, feat) in zip(calibrated_probs, feature_df.iterrows()):
            res = self.evaluate_record_pair(p, feat.to_dict())
            results.append(res)
        return results

    def compute_policy_metrics(self, calibrated_probs: np.ndarray, y_true: np.ndarray, feature_df: pd.DataFrame) -> Dict[str, float]:
        policy_results = self.predict_policy(calibrated_probs, feature_df)
        decisions = [r["decision"] for r in policy_results]

        n_total = len(y_true)
        n_match = sum(1 for d in decisions if d == DECISION_MATCH)
        n_non_match = sum(1 for d in decisions if d == DECISION_NON_MATCH)
        n_review = sum(1 for d in decisions if d == DECISION_REVIEW)

        n_auto = n_match + n_non_match
        coverage = n_auto / max(n_total, 1)
        review_rate = n_review / max(n_total, 1)

        auto_correct = 0
        for r, label in zip(policy_results, y_true):
            d = r["decision"]
            if d == DECISION_MATCH and label == 1:
                auto_correct += 1
            elif d == DECISION_NON_MATCH and label == 0:
                auto_correct += 1

        auto_precision = auto_correct / max(n_auto, 1)
        false_auto_rate = 1.0 - auto_precision if n_auto > 0 else 0.0

        return {
            "auto_match_precision": float(auto_precision),
            "auto_match_coverage": float(coverage),
            "review_rate": float(review_rate),
            "false_auto_match_rate": float(false_auto_rate),
            "count_auto": n_auto,
            "count_review": n_review,
            "count_total": n_total,
        }
