import pytest
from fastapi.testclient import TestClient
from switchrank.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"

def test_match_endpoint_match():
    payload = {
        "record_left": {
            "title": "BD Syringe 10 mL Luer-Lok Tip",
            "brand": "BD",
            "description": "Sterile piston syringe 10ml"
        },
        "record_right": {
            "title": "BD 10ml Luer Lock Syringe Ref 309604",
            "brand": "BD",
            "description": "10 mL syringe piston single-use"
        }
    }
    response = client.post("/match", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] in ["MATCH", "REVIEW", "NON_MATCH"]
    assert "calibrated_confidence" in data
    assert isinstance(data["supporting_evidence"], list)

def test_match_endpoint_numeric_conflict():
    payload = {
        "record_left": {
            "title": "Sandisk Cruzer 16 GB Flash Drive",
            "brand": "Sandisk"
        },
        "record_right": {
            "title": "Sandisk Cruzer 64 GB Flash Drive",
            "brand": "Sandisk"
        }
    }
    response = client.post("/match", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "REVIEW"
    assert "numeric specifications: mismatch detected" in data["conflicting_evidence"]
