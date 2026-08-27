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

import numpy as np
import pandas as pd

WINDOW = "1min"


def _historical_stats(train_df: pd.DataFrame):
    """Compute baseline statistics from TRAIN set only."""
    normal_train = train_df[train_df.is_fraud_spike == 0]
    return {
        "amount_mean": normal_train.amount.mean(),
        "amount_std": normal_train.amount.std(),
        "known_devices": set(normal_train.device_id.unique()),
        "known_geos": set(normal_train.geo.unique()),
    }


def build_window_features(df: pd.DataFrame, hist_stats: dict) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
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

    # Drop empty windows
    feats = feats[feats["txn_count"] > 0].copy()

    # Derived features
    feats["device_reuse_rate"] = feats["txn_count"] / feats["unique_devices"]
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

    return feats


def main():
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
