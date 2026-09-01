"""
FastAPI REST server for real-time fraud-spike scoring.

Demonstrates production readiness: trained models are loaded from disk and
used to score incoming transaction batches in real time.

Endpoints:
  POST /score      — Score a batch of transactions
  GET  /health     — Health check
  GET  /models     — List loaded models with metadata
  GET  /detectors  — List available detector names
  GET  /features   — List all 16 engineered features with descriptions
"""

from __future__ import annotations

import os
import sys
import json
import logging
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_store import load_all_models
from models import FEATURE_COLS
from features import build_window_features, _historical_stats

log = logging.getLogger(__name__)

# Application state
_state: dict[str, Any] = {
    "detectors": [],
    "hist_stats": None,
    "metadata": {},
    "startup_time": None,
}


# --- Lifespan (replaces deprecated @app.on_event) ---

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models and historical stats at startup, clean up on shutdown."""
    log.info("Starting SentinelRisk-AI API server...")

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    if os.path.exists(model_dir):
        _state["detectors"] = load_all_models(model_dir)
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                _state["metadata"] = json.load(f)

    train_path = os.path.join(data_dir, "transactions_train.csv")
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        _state["hist_stats"] = _historical_stats(train_df)

    _state["startup_time"] = datetime.now().isoformat()
    log.info("Loaded %d detectors, ready to serve.", len(_state["detectors"]))

    yield  # Server is running

    log.info("Shutting down SentinelRisk-AI API server.")


# --- App ---

app = FastAPI(
    title="SentinelRisk-AI REST Engine",
    description="Production-grade real-time fraud-spike detection API for digital payment streams. "
                "Leverages multi-model ensembles, vectorized feature engineering, and explainable AI reason codes.",
    version="2.1.0",
    lifespan=lifespan,
)

# CORS middleware — allows cross-origin requests from dashboards and frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response logging middleware ---

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request method, path, and response time."""
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    log.info("%s %s → %d (%.3fs)", request.method, request.url.path,
             response.status_code, elapsed)
    return response


# --- Error handler ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return structured JSON errors."""
    log.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


# --- Models ---

class Transaction(BaseModel):
    txn_id: str = Field(..., description="Unique transaction identifier")
    timestamp: str = Field(..., description="Transaction timestamp (ISO format or 'YYYY-MM-DD HH:MM:SS')")
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    device_id: str = Field(..., description="Device identifier")
    geo: str = Field(..., description="Geographic location")


class ScoreRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, description="List of transactions to score")
    detector: str = Field("ensemble_weighted_vote", description="Detector to use for scoring")


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


# --- Feature descriptions for /features endpoint ---

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "txn_count": "Number of transactions in the 1-minute window",
    "unique_devices": "Count of distinct device IDs in the window",
    "amount_mean": "Mean transaction amount in the window",
    "amount_max": "Maximum transaction amount in the window",
    "amount_z_mean": "Mean z-score of transaction amounts (vs. merchant baseline)",
    "amount_z_max": "Max z-score of transaction amounts (vs. merchant baseline)",
    "new_device_ratio": "Fraction of transactions from previously unseen devices",
    "new_geo_ratio": "Fraction of transactions from previously unseen geos",
    "device_reuse_rate": "Ratio of txn_count / unique_devices (device concentration)",
    "amount_std": "Standard deviation of transaction amounts in the window",
    "amount_cv": "Coefficient of variation of amounts (std / mean)",
    "hour_of_day": "Hour of day (0-23) for temporal pattern detection",
    "is_weekend": "1 if the window falls on Saturday or Sunday, else 0",
    "geo_entropy": "Number of unique geographies in the window",
    "device_entropy": "Number of unique devices in the window",
    "amount_skewness": "Skewness of the amount distribution in the window",
}


# --- Endpoints ---

@app.get("/health")
def health() -> dict[str, Any]:
    """Health check with uptime and detector count."""
    return {
        "status": "ok",
        "detectors_loaded": len(_state["detectors"]),
        "detector_names": [d.name for d in _state["detectors"]],
        "startup_time": _state["startup_time"],
    }


@app.get("/models")
def models_info() -> dict[str, Any]:
    """Return model metadata including version, features, and detector details."""
    return _state["metadata"]


@app.get("/detectors")
def list_detectors() -> dict[str, Any]:
    """List all available detectors with their safe names for the /score endpoint."""
    detectors_info = []
    for d in _state["detectors"]:
        safe_name = d.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        detectors_info.append({
            "name": d.name,
            "safe_name": safe_name,
            "type": type(d).__name__,
        })
    return {"detectors": detectors_info}


@app.get("/features")
def list_features() -> dict[str, Any]:
    """List all 16 engineered features with human-readable descriptions."""
    return {
        "n_features": len(FEATURE_COLS),
        "features": [
            {"name": col, "description": FEATURE_DESCRIPTIONS.get(col, "")}
            for col in FEATURE_COLS
        ],
    }


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    """Score a batch of transactions using the specified detector."""
    if not _state["detectors"]:
        raise HTTPException(status_code=503, detail="No models loaded — run the pipeline first")
    if _state["hist_stats"] is None:
        raise HTTPException(status_code=503, detail="Historical stats not loaded — training data missing")

    # Find the requested detector
    detector = None
    for d in _state["detectors"]:
        safe = d.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if safe == request.detector or d.name == request.detector:
            detector = d
            break
    if detector is None:
        available = [d.name for d in _state["detectors"]]
        raise HTTPException(
            status_code=400,
            detail=f"Detector '{request.detector}' not found. Available: {available}"
        )

    # Convert transactions to DataFrame
    txn_data = [t.model_dump() for t in request.transactions]
    df = pd.DataFrame(txn_data)
    df["is_fraud_spike"] = 0  # unknown for incoming data

    # Build features
    feats = build_window_features(df, _state["hist_stats"])

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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
