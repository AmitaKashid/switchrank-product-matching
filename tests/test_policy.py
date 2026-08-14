import pytest
import pandas as pd
from switchrank.decision.policy import SelectiveDecisionPolicy, DECISION_MATCH, DECISION_REVIEW, DECISION_NON_MATCH

def test_decision_policy_routing():
    policy = SelectiveDecisionPolicy(match_threshold=0.85, non_match_threshold=0.20)

    # High confidence match
    res_match = policy.evaluate_record_pair(0.95, {"brand_exact": 1.0, "numeric_mismatch": 0.0})
    assert res_match["decision"] == DECISION_MATCH

    # Ambiguous confidence review
    res_amb = policy.evaluate_record_pair(0.50, {"numeric_mismatch": 0.0})
    assert res_amb["decision"] == DECISION_REVIEW

    # Low confidence non match
    res_non = policy.evaluate_record_pair(0.10, {"numeric_mismatch": 0.0})
    assert res_non["decision"] == DECISION_NON_MATCH

def test_numeric_conflict_override():
    policy = SelectiveDecisionPolicy(match_threshold=0.85, non_match_threshold=0.20)
    # Even if score is 0.96, numeric mismatch MUST route to REVIEW
    res = policy.evaluate_record_pair(0.96, {"numeric_mismatch": 1.0})
    assert res["decision"] == DECISION_REVIEW
    assert "numeric specifications: mismatch detected" in res["conflicting_evidence"]
