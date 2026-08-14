import os
import io
import urllib.request
import zipfile
import gzip
import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed/wdc")
WDC_URL = "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80pair.zip"

FILES_TO_EXTRACT = {
    "train": "wdcproducts80cc20rnd000un_train_small.json.gz",
    "valid": "wdcproducts80cc20rnd000un_valid_small.json.gz",
    "test_000un": "wdcproducts80cc20rnd000un_gs.json.gz",
    "test_050un": "wdcproducts80cc20rnd050un_gs.json.gz",
    "test_100un": "wdcproducts80cc20rnd100un_gs.json.gz",
}

def download_and_extract_wdc():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = RAW_DIR / "80pair.zip"
    if not zip_path.exists():
        print(f"Downloading WDC 80pair.zip from {WDC_URL}...")
        req = urllib.request.Request(WDC_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
            f.write(resp.read())
        print(f"Saved {zip_path} ({zip_path.stat().st_size} bytes)")
    else:
        print(f"Using existing {zip_path}")
        
    with zipfile.ZipFile(zip_path, "r") as zf:
        available_files = zf.namelist()
        print(f"Archive contains {len(available_files)} files.")
        
        for key, fname in FILES_TO_EXTRACT.items():
            if fname in available_files:
                print(f"Processing {fname} -> split: {key}")
                with zf.open(fname) as gz_file:
                    with gzip.GzipFile(fileobj=gz_file) as gz:
                        records = []
                        for line in gz:
                            line_str = line.decode("utf-8").strip()
                            if line_str:
                                records.append(json.loads(line_str))
                        
                        df = pd.DataFrame(records)
                        output_path = PROCESSED_DIR / f"{key}.csv"
                        df.to_csv(output_path, index=False)
                        print(f"Saved {output_path} ({len(df)} records)")
            else:
                print(f"Warning: {fname} not found in zip archive!")

if __name__ == "__main__":
    download_and_extract_wdc()
