"""
Feature engineering for fraud-spike detection (v2 — enhanced & vectorized).

We bucket transactions into fixed time windows (default: 1 minute) and compute,
for EACH window, features that separate normal traffic from fraud spikes:

  - txn_count            -> VELOCITY_BURST signal
  - unique_devices       -> Low unique-device count despite high volume
  - amount_mean/max      -> AMOUNT_ANOMALY signal
  - amount_z_mean/max    -> Amount z-scores relative to merchant baseline
  - new_device_ratio     -> GEO_DEVICE_CLUSTER signal
  - new_geo_ratio        -> Unfamiliar geos
  - device_reuse_rate    -> Same device used repeatedly
  - amount_std           -> Amount dispersion
  - amount_cv            -> Coefficient of variation
  - hour_of_day          -> Temporal signal
  - is_weekend           -> Day-of-week signal
  - geo_entropy          -> Unique geo count per window
  - device_entropy       -> Unique device count per window
  - amount_skewness      -> Amount distribution shape
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WINDOW = "1min"


def _historical_stats(train_df: pd.DataFrame) -> dict[str, Any]:
    """Compute baseline statistics from TRAIN set only.

    Returns a dict with keys: amount_mean, amount_std, known_devices, known_geos.
    """
    normal_train = train_df[train_df.is_fraud_spike == 0]
    if len(normal_train) == 0:
        log.warning("No normal transactions in training data — using full dataset for stats")
        normal_train = train_df

    amount_std = normal_train.amount.std()
    if amount_std == 0 or np.isnan(amount_std):
        amount_std = 1.0
        log.warning("Amount standard deviation is 0 or NaN — defaulting to 1.0")

    return {
        "amount_mean": float(normal_train.amount.mean()),
        "amount_std": float(amount_std),
        "known_devices": set(normal_train.device_id.unique()),
        "known_geos": set(normal_train.geo.unique()),
    }


def build_window_features(df: pd.DataFrame, hist_stats: dict[str, Any]) -> pd.DataFrame:
    """Build rolling-window features from a transaction DataFrame.

    Args:
        df: Transaction DataFrame with columns: timestamp, amount, device_id, geo,
            is_fraud_spike, and optionally spike_type.
        hist_stats: Historical statistics from _historical_stats().

    Returns:
        A DataFrame with one row per non-empty time window, containing all 16
        engineered features plus window_start and window_label.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if len(df) == 0:
        log.warning("Empty transaction DataFrame — returning empty features")
        return pd.DataFrame()

    df = df.set_index("timestamp").sort_index()

    df["is_known_device"] = df["device_id"].isin(hist_stats["known_devices"]).astype(float)
    df["is_known_geo"] = df["geo"].isin(hist_stats["known_geos"]).astype(float)
    df["amount_z"] = (df["amount"] - hist_stats["amount_mean"]) / hist_stats["amount_std"]

    grouped = df.resample(WINDOW)

    feats = pd.DataFrame({
        "txn_count": grouped["amount"].count(),
        "unique_devices": grouped["device_id"].nunique(),
        "amount_mean": grouped["amount"].mean(),
        "amount_max": grouped["amount"].max(),
        "amount_z_mean": grouped["amount_z"].mean(),
        "amount_z_max": grouped["amount_z"].max(),
        "new_device_ratio": 1.0 - grouped["is_known_device"].mean(),
        "new_geo_ratio": 1.0 - grouped["is_known_geo"].mean(),
        "window_label": grouped["is_fraud_spike"].max(),
        "amount_std": grouped["amount"].std(),
        "geo_entropy": grouped["geo"].nunique(),
        "device_entropy": grouped["device_id"].nunique(),
        "amount_skewness": grouped["amount"].skew(),
    })

    # Preserve dominant spike_type per window if available
    if "spike_type" in df.columns:
        feats["spike_type"] = grouped["spike_type"].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else "none"
        )

    # Drop empty windows
    feats = feats[feats["txn_count"] > 0].copy()

    if len(feats) == 0:
        log.warning("All windows are empty after filtering — returning empty features")
        return pd.DataFrame()

    # Derived features with safe division
    feats["device_reuse_rate"] = feats["txn_count"] / feats["unique_devices"].clip(lower=1)
    feats["amount_cv"] = feats["amount_std"] / (feats["amount_mean"].abs() + 1e-10)

    # Temporal features
    feats = feats.reset_index().rename(columns={"timestamp": "window_start"})
    feats["hour_of_day"] = feats["window_start"].dt.hour
    feats["is_weekend"] = feats["window_start"].dt.dayofweek.isin([5, 6]).astype(int)

    # Clean missing values
    feats["window_label"] = feats["window_label"].fillna(0).astype(int)
    feats["amount_std"] = feats["amount_std"].fillna(0)
    feats["amount_skewness"] = feats["amount_skewness"].fillna(0)
    feats = feats.fillna(0)

    log.info("Built %d window features (%d positive)", len(feats), int(feats["window_label"].sum()))
    return feats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train_raw = pd.read_csv("data/transactions_train.csv")
    test_raw = pd.read_csv("data/transactions_test.csv")

    hist_stats = _historical_stats(train_raw)

    train_feats = build_window_features(train_raw, hist_stats)
    test_feats = build_window_features(test_raw, hist_stats)

    train_feats.to_csv("data/features_train.csv", index=False)
    test_feats.to_csv("data/features_test.csv", index=False)

    print(f"Train windows: {len(train_feats)}  (positive: {train_feats.window_label.sum()})")
    print(f"Test windows:  {len(test_feats)}  (positive: {test_feats.window_label.sum()})")
    print(f"\nTrain positive rate: {train_feats.window_label.mean():.4f}")
    print(f"Test positive rate:  {test_feats.window_label.mean():.4f}")
    print(f"\nFeature columns ({len(train_feats.columns)}):")
    for col in train_feats.columns:
        print(f"  {col}")
    print("\nSample feature rows (positive windows):")
    print(train_feats[train_feats.window_label == 1].head(3).to_string())


if __name__ == "__main__":
    main()
