import re
import unicodedata
from typing import Dict, Any, Optional

UNIT_MAP = {
    r"(\d+)\s*ml\b": r"\1ml",
    r"(\d+)\s*cc\b": r"\1ml", # 1 cc = 1 ml in medical devices
    r"(\d+)\s*gb\b": r"\1gb",
    r"(\d+)\s*tb\b": r"\1tb",
    r"(\d+)\s*mm\b": r"\1mm",
    r"(\d+)\s*cm\b": r"\1cm",
    r"(\d+)\s*inch(es)?\b": r"\1in",
}

def normalize_text(text: Optional[str]) -> str:
    """Perform deterministic Unicode normalization, case normalization, and whitespace cleanup."""
    if not text or pd_isna(text):
        return ""

    # Unicode NFKD normalization
    text = unicodedata.normalize("NFKD", str(text))
    # Strip trademark/registered symbols
    text = text.replace("®", "").replace("™", "").replace("©", "")
    # Lowercase
    text = text.lower()
    # Normalize units
    for pattern, repl in UNIT_MAP.items():
        text = re.sub(pattern, repl, text)
    # Strip punctuation except alphanumeric and whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = " ".join(text.split())
    return text

def normalize_model_number(model_str: Optional[str]) -> str:
    """Extract and standardize model / MPN / catalog number tokens."""
    if not model_str or pd_isna(model_str):
        return ""
    raw = str(model_str).upper()
    # Strip separators (dashes, slashes, spaces) for canonical representation
    clean = re.sub(r"[^A-Z0-9]", "", raw)
    return clean

def pd_isna(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and str(val) == "nan":
        return True
    return False

class ProductNormalizer:
    """Deterministic product normalization pipeline preserving raw fields."""

    def normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        raw_title = record.get("title") or record.get("title_left") or record.get("title_right") or ""
        raw_brand = record.get("brand") or record.get("brand_left") or record.get("brand_right") or ""
        raw_desc = record.get("description") or record.get("description_left") or record.get("description_right") or ""

        # Extract numeric tokens from raw text before unit boundary changes
        numbers = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)", raw_title)]

        norm_title = normalize_text(raw_title)
        norm_brand = normalize_text(raw_brand)
        norm_desc = normalize_text(raw_desc)

        # Extract model numbers / catalog strings using regex patterns
        model_matches = re.findall(r"\b[A-Z0-9]{3,}(?:[\-\/][A-Z0-9]+)*\b", raw_title, re.IGNORECASE)
        norm_models = [normalize_model_number(m) for m in model_matches if len(normalize_model_number(m)) >= 3]

        return {
            "raw_title": raw_title,
            "raw_brand": raw_brand,
            "raw_description": raw_desc,
            "norm_title": norm_title,
            "norm_brand": norm_brand,
            "norm_description": norm_desc,
            "extracted_models": list(set(norm_models)),
            "extracted_numbers": numbers,
        }
