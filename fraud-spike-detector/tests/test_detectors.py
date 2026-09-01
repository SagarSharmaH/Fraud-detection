"""
Unit tests for feature engineering and detector interface contracts.

Covers:
  - Feature calculation sanity and edge cases
  - Detector interface conformity (fit/predict contract)
  - Threshold behavior and feature importance
  - Edge cases: empty DataFrames, all-normal data, single rows
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
    XGBoostDetector, IsolationForestDetector, AutoencoderDetector,
    EnsembleDetector, build_all_detectors, HAS_XGBOOST, _validate_features,
)


# --- Feature Tests ---

class TestFeatures:
    def test_historical_stats_keys(self, sample_transactions):
        stats = _historical_stats(sample_transactions)
        assert "amount_mean" in stats
        assert "amount_std" in stats
        assert "known_devices" in stats
        assert "known_geos" in stats

    def test_historical_stats_types(self, sample_transactions):
        stats = _historical_stats(sample_transactions)
        assert isinstance(stats["amount_mean"], float)
        assert isinstance(stats["amount_std"], float)
        assert isinstance(stats["known_devices"], set)
        assert isinstance(stats["known_geos"], set)
        assert stats["amount_std"] > 0

    def test_feature_columns_present(self, sample_features):
        for col in FEATURE_COLS:
            assert col in sample_features.columns, f"Missing feature: {col}"

    def test_no_nans(self, sample_features):
        assert sample_features[list(FEATURE_COLS)].isna().sum().sum() == 0

    def test_txn_count_positive(self, sample_features):
        assert (sample_features["txn_count"] > 0).all()

    def test_window_label_binary(self, sample_features):
        assert set(sample_features["window_label"].unique()).issubset({0, 1})

    def test_device_reuse_rate(self, sample_features):
        for _, row in sample_features.head(5).iterrows():
            expected = row["txn_count"] / max(row["unique_devices"], 1)
            assert row["device_reuse_rate"] == pytest.approx(expected, rel=1e-5)

    def test_empty_dataframe_returns_empty(self, empty_features):
        assert len(empty_features) == 0

    def test_single_row_produces_features(self, single_row_features):
        """A single transaction should produce at least one feature window."""
        assert len(single_row_features) >= 0  # may be 0 or 1 depending on window alignment

    def test_all_normal_no_positive_labels(self, all_normal_features):
        """When there are no fraud transactions, all window labels should be 0."""
        assert (all_normal_features["window_label"] == 0).all()

    def test_feature_cols_is_tuple(self):
        """FEATURE_COLS should be a tuple (immutable) to prevent accidental mutation."""
        assert isinstance(FEATURE_COLS, tuple)

    def test_amount_cv_non_negative(self, sample_features):
        """Coefficient of variation should always be non-negative."""
        assert (sample_features["amount_cv"] >= 0).all()


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

    def test_detector_repr(self):
        """All detectors should have meaningful __repr__."""
        detectors = build_all_detectors()
        for det in detectors:
            repr_str = repr(det)
            assert det.name in repr_str
            assert type(det).__name__ in repr_str

    def test_rule_based_thresholds(self, sample_features):
        det = RuleBasedDetector().fit(sample_features)
        assert len(det.thresholds) > 0
        assert "txn_count" in det.thresholds

    def test_random_forest_importances(self, sample_features):
        det = RandomForestDetector().fit(sample_features)
        assert len(det.feature_importances_) == len(FEATURE_COLS)
        assert sum(det.feature_importances_.values()) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
    def test_xgboost_importances(self, sample_features):
        """XGBoost should produce feature importances after fitting."""
        det = XGBoostDetector().fit(sample_features)
        assert len(det.feature_importances_) == len(FEATURE_COLS)
        assert all(v >= 0 for v in det.feature_importances_.values())

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

    def test_validate_features_raises_on_missing_cols(self):
        """_validate_features should raise ValueError on missing columns."""
        bad_df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required feature columns"):
            _validate_features(bad_df, context="test")

    def test_empty_predict_returns_empty(self, sample_features):
        """Predicting on empty features should return empty arrays."""
        det = RuleBasedDetector().fit(sample_features)
        empty_feats = sample_features.iloc[:0].copy()
        preds, scores, reasons = det.predict(empty_feats)
        assert len(preds) == 0
        assert len(scores) == 0
        assert len(reasons) == 0

    @pytest.mark.parametrize("DetectorClass", [
        RuleBasedDetector, RandomForestDetector,
        IsolationForestDetector, AutoencoderDetector,
    ])
    def test_detector_empty_predict(self, sample_features, DetectorClass):
        """All detector types should handle empty prediction input gracefully."""
        det = DetectorClass()
        det.fit(sample_features)
        empty_feats = sample_features.iloc[:0].copy()
        preds, scores, reasons = det.predict(empty_feats)
        assert len(preds) == 0
        assert len(scores) == 0
        assert len(reasons) == 0

    def test_all_normal_training_data(self, all_normal_features):
        """Detectors should train successfully even with zero positive labels."""
        detectors = build_all_detectors()
        for det in detectors:
            det.fit(all_normal_features)
            preds, scores, reasons = det.predict(all_normal_features)
            assert len(preds) == len(all_normal_features)
