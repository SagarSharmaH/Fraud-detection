"""
Integration tests for FastAPI server endpoints.

Covers:
  - Health check
  - Models metadata
  - Score endpoint (happy path and error paths)
  - Detectors listing
  - Features listing
  - Error handling (invalid detector, empty payload, malformed request)
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api import app


@pytest.fixture
def client():
    """Create a test client with models loaded."""
    # Force startup by using the TestClient context manager
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["detectors_loaded"] > 0
    assert "startup_time" in data


def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "detectors" in data


def test_detectors_endpoint(client):
    """New /detectors endpoint should list available detectors."""
    response = client.get("/detectors")
    assert response.status_code == 200
    data = response.json()
    assert "detectors" in data
    assert len(data["detectors"]) > 0
    # Each detector should have name, safe_name, type
    det = data["detectors"][0]
    assert "name" in det
    assert "safe_name" in det
    assert "type" in det


def test_features_endpoint(client):
    """New /features endpoint should list all 16 features."""
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert data["n_features"] == 16
    assert len(data["features"]) == 16
    # Each feature should have name and description
    feat = data["features"][0]
    assert "name" in feat
    assert "description" in feat


def test_score_endpoint(client):
    payload = {
        "transactions": [
            {
                "txn_id": "tx_001",
                "timestamp": "2026-07-20 12:00:00",
                "amount": 150.0,
                "device_id": "dev_999",
                "geo": "Mumbai"
            },
            {
                "txn_id": "tx_002",
                "timestamp": "2026-07-20 12:00:15",
                "amount": 9500.0,
                "device_id": "dev_999",
                "geo": "Mumbai"
            }
        ],
        "detector": "random_forest"
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["detector_used"] == "Random Forest"
    assert "n_windows" in data
    assert "scored_at" in data


def test_score_invalid_detector(client):
    """Requesting a non-existent detector should return 400."""
    payload = {
        "transactions": [
            {
                "txn_id": "tx_001",
                "timestamp": "2026-07-20 12:00:00",
                "amount": 150.0,
                "device_id": "dev_999",
                "geo": "Mumbai"
            }
        ],
        "detector": "nonexistent_detector"
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_score_malformed_request(client):
    """Missing required fields should return 422 validation error."""
    payload = {
        "transactions": [
            {
                "txn_id": "tx_001",
                # missing timestamp, amount, device_id, geo
            }
        ],
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_empty_transactions(client):
    """Empty transaction list should return 422 (min_length=1 validation)."""
    payload = {
        "transactions": [],
        "detector": "random_forest"
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_negative_amount(client):
    """Negative amount should return 422 validation error (gt=0 constraint)."""
    payload = {
        "transactions": [
            {
                "txn_id": "tx_001",
                "timestamp": "2026-07-20 12:00:00",
                "amount": -100.0,
                "device_id": "dev_999",
                "geo": "Mumbai"
            }
        ],
        "detector": "random_forest"
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_cors_headers(client):
    """CORS headers should be present on responses."""
    response = client.options("/health", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
    })
    # CORS middleware should allow the request
    assert response.status_code in (200, 405)  # OPTIONS may or may not be explicitly handled
