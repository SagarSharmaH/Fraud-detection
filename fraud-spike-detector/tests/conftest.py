"""
Shared test fixtures for the SentinelRisk-AI test suite.

Centralizes sample data generation and trained detector creation so
individual test modules don't duplicate setup logic.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_transactions():
    """Generate a small synthetic transaction DataFrame for testing."""
    rng = np.random.default_rng(123)
    n = 200
    timestamps = pd.date_range("2026-06-01", periods=n, freq="30s")
    return pd.DataFrame({
        "txn_id": [f"t{i:04d}" for i in range(n)],
        "timestamp": timestamps,
        "amount": rng.lognormal(6.5, 0.8, n).round(2),
        "device_id": rng.choice([f"dev_{i:03d}" for i in range(20)], n),
        "geo": rng.choice(["Mumbai", "Delhi", "Bangalore"], n),
        "is_fraud_spike": np.concatenate([np.zeros(180), np.ones(20)]).astype(int),
        "spike_type": ["none"] * 180 + ["VELOCITY_BURST"] * 20,
    })


@pytest.fixture
def sample_features(sample_transactions):
    """Build features from sample transactions."""
    from features import build_window_features, _historical_stats
    hist = _historical_stats(sample_transactions)
    return build_window_features(sample_transactions, hist)


@pytest.fixture
def trained_detectors(sample_features):
    """Return a list of all detectors, already fitted on sample features."""
    from models import build_all_detectors, EnsembleDetector
    detectors = build_all_detectors()
    for det in detectors:
        det.fit(sample_features)
    ensemble = EnsembleDetector(detectors=list(detectors))
    ensemble.fit(sample_features)
    detectors.append(ensemble)
    return detectors


@pytest.fixture
def empty_features(sample_transactions):
    """Build features from an empty transaction DataFrame (edge case)."""
    from features import build_window_features, _historical_stats
    hist = _historical_stats(sample_transactions)
    empty_df = sample_transactions.iloc[:0].copy()
    return build_window_features(empty_df, hist)


@pytest.fixture
def single_row_features(sample_transactions):
    """Build features from a single transaction (edge case)."""
    from features import build_window_features, _historical_stats
    hist = _historical_stats(sample_transactions)
    single_df = sample_transactions.iloc[:1].copy()
    return build_window_features(single_df, hist)


@pytest.fixture
def all_normal_features(sample_transactions):
    """Features from an all-normal (no fraud) transaction set."""
    from features import build_window_features, _historical_stats
    normal_df = sample_transactions.copy()
    normal_df["is_fraud_spike"] = 0
    hist = _historical_stats(normal_df)
    return build_window_features(normal_df, hist)
