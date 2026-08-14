import pytest
from switchrank.features.extractor import PairFeatureExtractor

def test_feature_extractor_pair():
    extractor = PairFeatureExtractor()
    r1 = {
        "title": "BD Syringe 10 mL Luer-Lok",
        "brand": "BD",
        "description": "Sterile piston syringe 10ml"
    }
    r2 = {
        "title": "BD 10ml Luer Lock Syringe",
        "brand": "BD",
        "description": "10 mL syringe piston single-use"
    }
    feat = extractor.extract_pair_features(r1, r2)
    assert feat["brand_exact"] == 1.0
    assert feat["title_token_overlap"] > 0.5
    assert feat["numeric_token_match"] == 1.0
    assert feat["numeric_mismatch"] == 0.0

def test_numeric_mismatch_feature():
    extractor = PairFeatureExtractor()
    r1 = {"title": "Sandisk 16 GB Flash Drive", "brand": "Sandisk"}
    r2 = {"title": "Sandisk 64 GB Flash Drive", "brand": "Sandisk"}
    feat = extractor.extract_pair_features(r1, r2)
    assert feat["numeric_mismatch"] == 1.0
    assert feat["numeric_token_match"] == 0.0
