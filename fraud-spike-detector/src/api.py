"""
FastAPI REST server for real-time fraud-spike scoring.

Demonstrates production readiness: trained models are loaded from disk and
used to score incoming transaction batches in real time.

Endpoints:
  POST /score   — Score a batch of transactions
  GET  /health  — Health check
  GET  /models  — List loaded models with metadata
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_store import load_all_models
from models import FEATURE_COLS
from features import build_window_features, _historical_stats

app = FastAPI(
    title="SentinelRisk-AI REST Engine",
    description="Production-grade real-time fraud-spike detection API for digital payment streams. "
                "Leverages multi-model ensembles, vectorized feature engineering, and explainable AI reason codes.",
    version="2.0.0",
)

# Globals loaded at startup
_detectors = []
_hist_stats = None
_metadata = {}


class Transaction(BaseModel):
    txn_id: str
    timestamp: str
    amount: float
    device_id: str
    geo: str


class ScoreRequest(BaseModel):
    transactions: list[Transaction]
    detector: str = "ensemble_weighted_vote"


class FlaggedWindow(BaseModel):
    window_start: str
    txn_count: int
    predicted_label: int
    score: float
    reasons: list[str]


class ScoreResponse(BaseModel):
    detector_used: str
    n_windows: int
    n_flagged: int
    flagged_windows: list[FlaggedWindow]
    scored_at: str


@app.on_event("startup")
def startup():
    global _detectors, _hist_stats, _metadata

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    if os.path.exists(model_dir):
        _detectors = load_all_models(model_dir)
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                _metadata = json.load(f)

    train_path = os.path.join(data_dir, "transactions_train.csv")
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        _hist_stats = _historical_stats(train_df)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "detectors_loaded": len(_detectors),
        "detector_names": [d.name for d in _detectors],
    }


@app.get("/models")
def models_info():
    return _metadata


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    if not _detectors:
        raise HTTPException(status_code=503, detail="No models loaded")
    if _hist_stats is None:
        raise HTTPException(status_code=503, detail="Historical stats not loaded")

    # Find the requested detector
    detector = None
    for d in _detectors:
        safe = d.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if safe == request.detector or d.name == request.detector:
            detector = d
            break
    if detector is None:
        available = [d.name for d in _detectors]
        raise HTTPException(
            status_code=400,
            detail=f"Detector '{request.detector}' not found. Available: {available}"
        )

    # Convert transactions to DataFrame
    txn_data = [t.model_dump() for t in request.transactions]
    df = pd.DataFrame(txn_data)
    df["is_fraud_spike"] = 0  # unknown for incoming data

    # Build features
    feats = build_window_features(df, _hist_stats)

    if len(feats) == 0:
        return ScoreResponse(
            detector_used=detector.name,
            n_windows=0,
            n_flagged=0,
            flagged_windows=[],
            scored_at=datetime.now().isoformat(),
        )

    # Score
    preds, scores, reasons = detector.predict(feats)

    flagged = []
    for i in range(len(feats)):
        if preds[i] == 1:
            flagged.append(FlaggedWindow(
                window_start=str(feats.iloc[i]["window_start"]),
                txn_count=int(feats.iloc[i]["txn_count"]),
                predicted_label=int(preds[i]),
                score=float(scores[i]),
                reasons=reasons[i],
            ))

    return ScoreResponse(
        detector_used=detector.name,
        n_windows=len(feats),
        n_flagged=int(preds.sum()),
        flagged_windows=flagged,
        scored_at=datetime.now().isoformat(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
