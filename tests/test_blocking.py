import pytest
from switchrank.blocking.blocker import CandidateBlocker

def test_brand_blocking():
    blocker = CandidateBlocker()
    records = [
        {"id": "1", "title": "Sandisk 16GB Flash Drive", "brand": "Sandisk"},
        {"id": "2", "title": "Sandisk 64GB USB Stick", "brand": "Sandisk"},
        {"id": "3", "title": "Kingston 32GB Flash Drive", "brand": "Kingston"},
    ]
    pairs = blocker.block_by_brand(records)
    assert ("1", "2") in pairs or ("2", "1") in pairs
    assert ("1", "3") not in pairs

def test_prefix_blocking():
    blocker = CandidateBlocker(prefix_len=4)
    records = [
        {"id": "101", "title": "Cruzer Force 16GB", "brand": ""},
        {"id": "102", "title": "Cruzer Glide 64GB", "brand": ""},
        {"id": "103", "title": "Extreme PRO 128GB", "brand": ""},
    ]
    pairs = blocker.block_by_prefix(records)
    assert ("101", "102") in pairs

def test_blocking_evaluation():
    blocker = CandidateBlocker()
    candidate_pairs = {("1", "2"), ("3", "4")}
    ground_truth = {("1", "2"), ("5", "6")}
    metrics = blocker.evaluate_blocking(candidate_pairs, ground_truth, total_universe_size=10)
    assert metrics["pair_completeness_recall"] == 0.5
    assert metrics["pair_reduction_ratio"] > 0.90
