from typing import Dict, Any, List
import pandas as pd
import numpy as np
from rapidfuzz import distance, fuzz
from switchrank.normalize.cleaner import ProductNormalizer, pd_isna

FEATURE_NAMES = [
    "brand_exact",
    "brand_sim",
    "brand_missing",
    "title_jaro_winkler",
    "title_token_set_ratio",
    "title_token_sort_ratio",
    "title_token_overlap",
    "description_sim",
    "description_token_overlap",
    "model_exact_match",
    "model_partial_match",
    "numeric_token_match",
    "numeric_mismatch",
    "length_ratio",
]

class PairFeatureExtractor:
    """Interpretable pairwise similarity feature extractor for product records."""

    def __init__(self):
        self.normalizer = ProductNormalizer()

    def extract_pair_features(self, record_left: Dict[str, Any], record_right: Dict[str, Any]) -> Dict[str, float]:
        norm_left = self.normalizer.normalize_record(record_left)
        norm_right = self.normalizer.normalize_record(record_right)

        b_left = norm_left["norm_brand"]
        b_right = norm_right["norm_brand"]

        # Brand agreement
        brand_missing = 1.0 if (not b_left or not b_right) else 0.0
        brand_exact = 1.0 if (b_left and b_right and b_left == b_right) else 0.0
        brand_sim = distance.JaroWinkler.similarity(b_left, b_right) if (b_left and b_right) else 0.0

        # Title similarity
        t_left = norm_left["norm_title"]
        t_right = norm_right["norm_title"]

        t_jw = distance.JaroWinkler.similarity(t_left, t_right) if (t_left and t_right) else 0.0
        t_set_ratio = fuzz.token_set_ratio(t_left, t_right) / 100.0 if (t_left and t_right) else 0.0
        t_sort_ratio = fuzz.token_sort_ratio(t_left, t_right) / 100.0 if (t_left and t_right) else 0.0

        tokens_l = set(t_left.split())
        tokens_r = set(t_right.split())
        if tokens_l and tokens_r:
            overlap = len(tokens_l.intersection(tokens_r)) / min(len(tokens_l), len(tokens_r))
        else:
            overlap = 0.0

        # Description similarity
        d_left = norm_left["norm_description"]
        d_right = norm_right["norm_description"]
        d_jw = distance.JaroWinkler.similarity(d_left, d_right) if (d_left and d_right) else 0.0
        dtokens_l = set(d_left.split())
        dtokens_r = set(d_right.split())
        if dtokens_l and dtokens_r:
            d_overlap = len(dtokens_l.intersection(dtokens_r)) / min(len(dtokens_l), len(dtokens_r))
        else:
            d_overlap = 0.0

        # Model number agreement
        m_left = norm_left["extracted_models"]
        m_right = norm_right["extracted_models"]
        model_exact = 1.0 if (m_left and m_right and bool(set(m_left).intersection(set(m_right)))) else 0.0

        if m_left and m_right:
            max_sim = 0.0
            for m1 in m_left:
                for m2 in m_right:
                    sim = distance.JaroWinkler.similarity(m1, m2)
                    if sim > max_sim:
                        max_sim = sim
            model_partial = max_sim
        else:
            model_partial = 0.0

        # Numeric token agreement
        n_left = norm_left["extracted_numbers"]
        n_right = norm_right["extracted_numbers"]
        if n_left and n_right:
            common_nums = set(n_left).intersection(set(n_right))
            numeric_match = 1.0 if common_nums else 0.0
            numeric_mismatch = 1.0 if (not common_nums and n_left != n_right) else 0.0
        else:
            numeric_match = 0.0
            numeric_mismatch = 0.0

        # Length ratio
        l1, l2 = len(t_left), len(t_right)
        len_ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 1.0

        return {
            "brand_exact": brand_exact,
            "brand_sim": brand_sim,
            "brand_missing": brand_missing,
            "title_jaro_winkler": t_jw,
            "title_token_set_ratio": t_set_ratio,
            "title_token_sort_ratio": t_sort_ratio,
            "title_token_overlap": overlap,
            "description_sim": d_jw,
            "description_token_overlap": d_overlap,
            "model_exact_match": model_exact,
            "model_partial_match": model_partial,
            "numeric_token_match": numeric_match,
            "numeric_mismatch": numeric_mismatch,
            "length_ratio": len_ratio,
        }

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        features_list = []
        for _, row in df.iterrows():
            left_rec = {
                "title": row.get("title_left"),
                "brand": row.get("brand_left"),
                "description": row.get("description_left"),
            }
            right_rec = {
                "title": row.get("title_right"),
                "brand": row.get("brand_right"),
                "description": row.get("description_right"),
            }
            feat = self.extract_pair_features(left_rec, right_rec)
            features_list.append(feat)

        feat_df = pd.DataFrame(features_list)
        return feat_df
