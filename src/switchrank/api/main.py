from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from switchrank.features.extractor import PairFeatureExtractor
from switchrank.models.supervised import SupervisedLightGBMMatcher
from switchrank.calibration.calibrator import ProbabilityCalibrator
from switchrank.decision.policy import SelectiveDecisionPolicy, DECISION_MATCH, DECISION_REVIEW, DECISION_NON_MATCH

app = FastAPI(
    title="SwitchRank API",
    description="Reliable Cross-Catalog Product Matching with Calibrated Abstention Decisions and Evidence Routing",
    version="0.1.0",
)

# Global pipeline state initialized lazily or on startup
feature_extractor = PairFeatureExtractor()
model_matcher = SupervisedLightGBMMatcher()
calibrator = ProbabilityCalibrator(method="isotonic")
decision_policy = SelectiveDecisionPolicy(match_threshold=0.80, non_match_threshold=0.25)
is_trained = False

def initialize_dummy_pipeline():
    """Fallback dummy training to ensure API functions standalone even before full experiment run."""
    global is_trained
    if is_trained:
        return
    dummy_data = pd.DataFrame([
        {"title_left": "Sandisk 64GB Flash Drive", "brand_left": "Sandisk", "description_left": "", "title_right": "Sandisk 64 GB USB Drive", "brand_right": "Sandisk", "description_right": "", "label": 1},
        {"title_left": "Sandisk 64GB Flash Drive", "brand_left": "Sandisk", "description_left": "", "title_right": "Sandisk 128 GB USB Drive", "brand_right": "Sandisk", "description_right": "", "label": 0},
        {"title_left": "BD 10mL Syringe Luer Lock", "brand_left": "BD", "description_left": "", "title_right": "BD 10ml Luer Lock Syringe", "brand_right": "BD", "description_right": "", "label": 1},
    ])
    model_matcher.fit(dummy_data)
    probs = model_matcher.predict_proba(dummy_data)
    calibrator.fit(probs, dummy_data["label"].values)
    is_trained = True

class ProductRecord(BaseModel):
    title: str = Field(..., example="BD Syringe 10 mL Luer-Lok Tip")
    brand: Optional[str] = Field(None, example="BD")
    description: Optional[str] = Field(None, example="Sterile piston syringe 10ml")

class PairMatchRequest(BaseModel):
    record_left: ProductRecord
    record_right: ProductRecord

class PairMatchResponse(BaseModel):
    decision: str = Field(..., example="MATCH")
    calibrated_confidence: float = Field(..., example=0.962)
    supporting_evidence: List[str]
    conflicting_evidence: List[str]
    review_reasons: List[str]

@app.on_event("startup")
def startup_event():
    initialize_dummy_pipeline()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SwitchRank Product Matching Engine",
        "version": "0.1.0",
        "pipeline_ready": is_trained,
    }

@app.post("/match", response_model=PairMatchResponse)
def match_records(request: PairMatchRequest):
    if not is_trained:
        initialize_dummy_pipeline()

    rec_left = request.record_left.dict()
    rec_right = request.record_right.dict()

    feat_dict = feature_extractor.extract_pair_features(rec_left, rec_right)
    df_single = pd.DataFrame([{
        "title_left": rec_left.get("title"),
        "brand_left": rec_left.get("brand"),
        "description_left": rec_left.get("description"),
        "title_right": rec_right.get("title"),
        "brand_right": rec_right.get("brand"),
        "description_right": rec_right.get("description"),
    }])

    raw_prob = float(model_matcher.predict_proba(df_single)[0])
    cal_prob = float(calibrator.calibrate(np.array([raw_prob]))[0])

    policy_res = decision_policy.evaluate_record_pair(cal_prob, feat_dict)

    return PairMatchResponse(
        decision=policy_res["decision"],
        calibrated_confidence=round(policy_res["calibrated_confidence"], 4),
        supporting_evidence=policy_res["supporting_evidence"],
        conflicting_evidence=policy_res["conflicting_evidence"],
        review_reasons=policy_res["review_reasons"],
    )
