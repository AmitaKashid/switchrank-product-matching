import os
import json
import urllib.request
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed/gudid")
FDA_URL = "https://api.fda.gov/device/510k.json?search=product_code:FMF&limit=1000"

def download_gudid_fmf():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_path = RAW_DIR / "gudid_fmf_raw.json"
    if not raw_path.exists():
        print(f"Downloading FDA 510(k) product code FMF data from openFDA API...")
        req = urllib.request.Request(FDA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(content)
        print(f"Saved raw response to {raw_path}")
    else:
        print(f"Using existing raw file {raw_path}")
        
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    results = data.get("results", [])
    print(f"Fetched {len(results)} FDA 510(k) records for product code FMF (Syringe, Piston).")
    
    clean_records = []
    for r in results:
        clean_records.append({
            "k_number": r.get("k_number"),
            "device_name": r.get("device_name"),
            "applicant": r.get("applicant"),
            "city": r.get("city"),
            "state": r.get("state"),
            "country_code": r.get("country_code"),
            "decision_date": r.get("decision_date"),
            "decision_description": r.get("decision_description"),
            "product_code": r.get("product_code", "FMF"),
        })
        
    df = pd.DataFrame(clean_records)
    output_path = PROCESSED_DIR / "fmf_canonical.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved canonical medical device catalog to {output_path} ({len(df)} records)")

if __name__ == "__main__":
    download_gudid_fmf()
