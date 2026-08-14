import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from switchrank.blocking.blocker import CandidateBlocker

PROCESSED_WDC = Path("data/processed/wdc")
REPORTS_DIR = Path("reports")

def test_data_leakage_safety():
    """Verify 0 pair ID overlap between train, valid, and all test splits."""
    train_df = pd.read_csv(PROCESSED_WDC / "train.csv")
    valid_df = pd.read_csv(PROCESSED_WDC / "valid.csv")
    test_000 = pd.read_csv(PROCESSED_WDC / "test_000un.csv")
    test_050 = pd.read_csv(PROCESSED_WDC / "test_050un.csv")
    test_100 = pd.read_csv(PROCESSED_WDC / "test_100un.csv")

    train_pairs = set(train_df["pair_id"].dropna())
    valid_pairs = set(valid_df["pair_id"].dropna())
    t0_pairs = set(test_000["pair_id"].dropna())
    t5_pairs = set(test_050["pair_id"].dropna())
    t1_pairs = set(test_100["pair_id"].dropna())

    assert len(train_pairs.intersection(valid_pairs)) == 0
    assert len(train_pairs.intersection(t0_pairs)) == 0
    assert len(valid_pairs.intersection(t0_pairs)) == 0
    assert len(train_pairs.intersection(t1_pairs)) == 0

def test_val_calib_val_policy_split_disjoint():
    """Verify validation_calibration and validation_policy splits are 100% disjoint."""
    valid_df = pd.read_csv(PROCESSED_WDC / "valid.csv")
    val_calib, val_policy = train_test_split(
        valid_df, test_size=0.5, random_state=42, stratify=valid_df["label"]
    )
    calib_pairs = set(val_calib["pair_id"].dropna())
    policy_pairs = set(val_policy["pair_id"].dropna())

    assert len(calib_pairs.intersection(policy_pairs)) == 0
    assert len(val_calib) + len(val_policy) == len(valid_df)

def test_blocking_metrics_computed_dynamically():
    """Verify CandidateBlocker dynamically computes pair reduction and recall."""
    blocker = CandidateBlocker(prefix_len=4, window_size=5)
    dummy_records = [
        {"id": "1", "title": "BD Syringe 10ml", "brand": "BD"},
        {"id": "2", "title": "BD 10ml Syringe", "brand": "BD"},
        {"id": "3", "title": "Terumo Syringe 5ml", "brand": "Terumo"},
    ]
    candidates = blocker.block_by_brand(dummy_records)
    eval_res = blocker.evaluate_blocking(candidates, {("1", "2")}, len(dummy_records))

    assert "candidate_count" in eval_res
    assert "total_possible" in eval_res
    assert "pair_reduction_ratio" in eval_res
    assert "pair_completeness_recall" in eval_res
    assert eval_res["total_possible"] == 3
    assert 0.0 <= eval_res["pair_reduction_ratio"] <= 1.0

def test_reproducible_metric_generation_artifact():
    """Verify authoritative final_metrics.json artifact exists and contains all required keys."""
    metrics_path = REPORTS_DIR / "final_metrics.json"
    assert metrics_path.exists(), "final_metrics.json does not exist"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = [
        "dataset",
        "blocking",
        "rapidfuzz",
        "rules",
        "fellegi_sunter",
        "lightgbm_standard",
        "lightgbm_hard_negative",
        "hard_negative_causal_delta",
        "calibration",
        "selective_policy",
        "healthcare_transfer",
    ]
    for key in required_keys:
        assert key in data, f"Missing required key in final_metrics.json: {key}"
