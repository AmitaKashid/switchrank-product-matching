import os
import random
import re
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_WDC = Path("data/processed/wdc")
PROCESSED_GUDID = Path("data/processed/gudid")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def verify_leakage_safety():
    print("=== Verifying WDC Train / Validation / Test Leakage Safety ===")
    train_df = pd.read_csv(PROCESSED_WDC / "train.csv")
    valid_df = pd.read_csv(PROCESSED_WDC / "valid.csv")
    test_000 = pd.read_csv(PROCESSED_WDC / "test_000un.csv")
    test_050 = pd.read_csv(PROCESSED_WDC / "test_050un.csv")
    test_100 = pd.read_csv(PROCESSED_WDC / "test_100un.csv")

    train_pairs = set(train_df["pair_id"].dropna())
    valid_pairs = set(valid_df["pair_id"].dropna())
    test_000_pairs = set(test_000["pair_id"].dropna())

    tv_overlap = train_pairs.intersection(valid_pairs)
    tt_overlap = train_pairs.intersection(test_000_pairs)

    print(f"Train pairs: {len(train_pairs)} | Valid pairs: {len(valid_pairs)} | Test 000un pairs: {len(test_000_pairs)}")
    print(f"Train/Valid Pair Overlap: {len(tv_overlap)}")
    print(f"Train/Test Pair Overlap: {len(tt_overlap)}")
    assert len(tv_overlap) == 0, "Leakage detected between Train and Validation sets!"
    assert len(tt_overlap) == 0, "Leakage detected between Train and Test sets!"
    print("PASS: Zero pair leakage verified across splits.")

def perturb_record(record: dict, difficulty: str) -> dict:
    device_name = str(record.get("device_name", ""))
    applicant = str(record.get("applicant", ""))
    k_number = str(record.get("k_number", ""))
    city = str(record.get("city", ""))

    title = f"{applicant} {device_name} (Model/Catalog {k_number})"

    if difficulty == "EASY":
        # Case, whitespace, unit formatting, punctuation
        title = title.lower() if random.random() < 0.5 else title.upper()
        title = title.replace("mL", "ml").replace("SYRINGE", "syringe")
        title = re.sub(r"[\.,\-]", " ", title)
        title = " ".join(title.split())
        brand = applicant.strip().lower()

    elif difficulty == "MEDIUM":
        # Abbreviate applicant, add packaging noise, strip trademark symbols
        words = applicant.split()
        short_app = words[0] if len(words) > 0 else applicant
        title = f"{short_app} - {device_name} [Ref #{k_number}] Pack of 100"
        title = title.replace("®", "").replace("™", "")
        if random.random() < 0.5:
            title = title.replace("Syringe", "Syr.").replace("Piston", "Pist.")
        brand = short_app

    elif difficulty == "HARD":
        # Remove manufacturer or brand, shorten description, reorder tokens
        tokens = device_name.split()
        random.shuffle(tokens)
        shuffled_name = " ".join(tokens)
        if random.random() < 0.5:
            title = f"{shuffled_name} {k_number}" # no manufacturer
            brand = ""
        else:
            title = f"{applicant} {shuffled_name}" # no model number
            brand = applicant

    return {"title": title, "brand": brand}

def create_gudid_stress_test():
    print("=== Constructing Healthcare Domain Transfer Stress Test ===")
    canonical_path = PROCESSED_GUDID / "fmf_canonical.csv"
    df = pd.read_csv(canonical_path)
    print(f"Loaded {len(df)} canonical medical device records.")

    sample_size = min(300, len(df))
    sample_df = df.head(sample_size).copy()

    records = sample_df.to_dict(orient="records")
    stress_pairs = []

    for i, rec in enumerate(records):
        left_id = rec["k_number"]
        left_title = f"{rec['applicant']} {rec['device_name']} Ref:{rec['k_number']}"
        left_brand = str(rec["applicant"])

        # Generate Positive Pairs across EASY, MEDIUM, HARD
        for diff in ["EASY", "MEDIUM", "HARD"]:
            pert = perturb_record(rec, diff)
            stress_pairs.append({
                "pair_id": f"POS_{diff}_{left_id}_{i}",
                "difficulty": diff,
                "label": 1,
                "id_left": left_id,
                "title_left": left_title,
                "brand_left": left_brand,
                "id_right": left_id,
                "title_right": pert["title"],
                "brand_right": pert["brand"],
            })

        # Generate Negative Pair (with a different canonical device)
        neg_idx = (i + random.randint(1, sample_size - 1)) % sample_size
        neg_rec = records[neg_idx]
        neg_id = neg_rec["k_number"]
        neg_pert = perturb_record(neg_rec, "MEDIUM")
        stress_pairs.append({
            "pair_id": f"NEG_MEDIUM_{left_id}_{neg_id}",
            "difficulty": "MEDIUM",
            "label": 0,
            "id_left": left_id,
            "title_left": left_title,
            "brand_left": left_brand,
            "id_right": neg_id,
            "title_right": neg_pert["title"],
            "brand_right": neg_pert["brand"],
        })

    stress_df = pd.DataFrame(stress_pairs)
    out_path = PROCESSED_GUDID / "fmf_stress_test.csv"
    stress_df.to_csv(out_path, index=False)
    print(f"Saved Healthcare Stress Test to {out_path} ({len(stress_df)} evaluation pairs across EASY, MEDIUM, HARD levels).")

if __name__ == "__main__":
    verify_leakage_safety()
    create_gudid_stress_test()
