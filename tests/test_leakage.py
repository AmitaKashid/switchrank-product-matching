import pandas as pd
from pathlib import Path

PROCESSED_WDC = Path("data/processed/wdc")

def test_data_leakage_safety():
    train_df = pd.read_csv(PROCESSED_WDC / "train.csv")
    valid_df = pd.read_csv(PROCESSED_WDC / "valid.csv")
    test_df = pd.read_csv(PROCESSED_WDC / "test_000un.csv")

    train_pairs = set(train_df["pair_id"].dropna())
    valid_pairs = set(valid_df["pair_id"].dropna())
    test_pairs = set(test_df["pair_id"].dropna())

    assert len(train_pairs.intersection(valid_pairs)) == 0
    assert len(train_pairs.intersection(test_pairs)) == 0
    assert len(valid_pairs.intersection(test_pairs)) == 0
