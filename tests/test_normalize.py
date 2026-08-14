import pytest
from switchrank.normalize.cleaner import ProductNormalizer, normalize_text, normalize_model_number

def test_normalize_text_basic():
    raw = "SanDisk® Cruzer Force 16 GB USB 2.0 Drive - Chrome"
    norm = normalize_text(raw)
    assert "sandisk" in norm
    assert "®" not in norm
    assert "16gb" in norm

def test_unit_conversion():
    raw = "BD Syringe 10 mL Luer-Lok (10 cc)"
    norm = normalize_text(raw)
    assert "10ml" in norm

def test_model_number_extraction():
    model = normalize_model_number("REF #309-604/A")
    assert model == "REF309604A"

def test_product_normalizer():
    normalizer = ProductNormalizer()
    rec = {
        "title": "BD Syringe 10 mL Luer-Lok™",
        "brand": "Becton Dickinson®",
        "description": "Sterile piston syringe 10ml"
    }
    result = normalizer.normalize_record(rec)
    assert result["norm_brand"] == "becton dickinson"
    assert "10ml" in result["norm_title"]
    assert 10.0 in result["extracted_numbers"]
