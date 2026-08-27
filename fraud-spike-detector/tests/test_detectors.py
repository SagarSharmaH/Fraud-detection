"""
Unit tests for feature engineering and detector interface contracts.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import build_window_features, _historical_stats
from models import (
    FEATURE_COLS, RuleBasedDetector, RandomForestDetector,
    IsolationForestDetector, AutoencoderDetector, EnsembleDetector,
    build_all_detectors,
)


# --- Fixtures ---

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
    })


@pytest.fixture
def sample_features(sample_transactions):
    """Build features from sample transactions."""
    hist = _historical_stats(sample_transactions)
    return build_window_features(sample_transactions, hist)


# --- Feature Tests ---

class TestFeatures:
    def test_historical_stats_keys(self, sample_transactions):
        stats = _historical_stats(sample_transactions)
        assert "amount_mean" in stats
        assert "amount_std" in stats
        assert "known_devices" in stats
        assert "known_geos" in stats

    def test_feature_columns_present(self, sample_features):
        for col in FEATURE_COLS:
            assert col in sample_features.columns, f"Missing feature: {col}"

    def test_no_nans(self, sample_features):
        assert sample_features[FEATURE_COLS].isna().sum().sum() == 0

    def test_txn_count_positive(self, sample_features):
        assert (sample_features["txn_count"] > 0).all()

    def test_window_label_binary(self, sample_features):
        assert set(sample_features["window_label"].unique()).issubset({0, 1})

    def test_device_reuse_rate(self, sample_features):
        for _, row in sample_features.head(5).iterrows():
            expected = row["txn_count"] / row["unique_devices"]
            assert row["device_reuse_rate"] == pytest.approx(expected, rel=1e-5)


# --- Detector Tests ---

class TestDetectors:
    def test_build_all_detectors(self):
        detectors = build_all_detectors()
        assert len(detectors) >= 4

    def test_detector_interface(self, sample_features):
        """All detectors must conform to the BaseDetector interface."""
        detectors = build_all_detectors()
        for det in detectors:
            assert hasattr(det, "name")
            assert hasattr(det, "fit")
            assert hasattr(det, "predict")

            det.fit(sample_features)
            preds, scores, reasons = det.predict(sample_features)

            assert isinstance(preds, np.ndarray)
            assert isinstance(scores, np.ndarray)
            assert isinstance(reasons, list)
            assert len(preds) == len(sample_features)
            assert len(scores) == len(sample_features)
            assert len(reasons) == len(sample_features)
            assert set(np.unique(preds)).issubset({0, 1})

    def test_rule_based_thresholds(self, sample_features):
        det = RuleBasedDetector().fit(sample_features)
        assert len(det.thresholds) > 0
        assert "txn_count" in det.thresholds

    def test_random_forest_importances(self, sample_features):
        det = RandomForestDetector().fit(sample_features)
        assert len(det.feature_importances_) == len(FEATURE_COLS)
        assert sum(det.feature_importances_.values()) == pytest.approx(1.0, abs=0.01)

    def test_reasons_only_for_flagged(self, sample_features):
        """Reasons should be empty for non-flagged windows."""
        det = RandomForestDetector().fit(sample_features)
        preds, _, reasons = det.predict(sample_features)
        for i in range(len(preds)):
            if preds[i] == 0:
                assert reasons[i] == []

    def test_ensemble_weights_sum_to_one(self, sample_features):
        detectors = build_all_detectors()
        for d in detectors:
            d.fit(sample_features)
        ens = EnsembleDetector(detectors=detectors)
        ens.fit(sample_features)
        total = sum(ens.weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_autoencoder_threshold(self, sample_features):
        det = AutoencoderDetector().fit(sample_features)
        assert det.threshold is not None
        assert det.threshold > 0
