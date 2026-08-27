"""
Integration tests for FastAPI server endpoints.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api import app, startup


@pytest.fixture
def client():
    startup()
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["detectors_loaded"] > 0


def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "detectors" in data


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
