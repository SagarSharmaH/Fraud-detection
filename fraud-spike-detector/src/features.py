"""
Feature engineering for fraud-spike detection.

We bucket transactions into fixed time windows (default: 1 minute) and compute,
for EACH window, features that should separate normal traffic from the 3 spike
types we injected:

  - txn_count            -> catches VELOCITY_BURST (raw volume spike)
  - unique_devices        -> low unique-device count despite high volume is
                              suspicious (one bot device firing repeatedly)
  - amount_mean / amount_zscore
                          -> catches AMOUNT_ANOMALY (unusually large amounts
                             vs the merchant's historical distribution)
  - new_device_ratio      -> catches GEO_DEVICE_CLUSTER (devices never seen
                             before, concentrated in a short burst)
  - new_geo_ratio         -> catches GEO_DEVICE_CLUSTER (unfamiliar geos)
  - device_reuse_rate     -> catches VELOCITY_BURST (same device used
                             repeatedly = txn_count / unique_devices)

The label for a window is 1 if ANY transaction in it is a fraud-spike txn
(this makes it a window-level detection problem, which is closer to how a
real-time monitoring system would operate: it doesn't need to catch every
single fraudulent transaction, it needs to catch the WINDOW/EVENT).
"""

import numpy as np
import pandas as pd

WINDOW = "1min"


def _historical_stats(train_df: pd.DataFrame):
    """Compute the merchant's baseline amount distribution and known
    devices/geos from the TRAIN set only. Using train-only stats to build
    features on the test set avoids leaking future information."""
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

    df["is_known_device"] = df["device_id"].isin(hist_stats["known_devices"])
    df["is_known_geo"] = df["geo"].isin(hist_stats["known_geos"])
    df["amount_z"] = (df["amount"] - hist_stats["amount_mean"]) / hist_stats["amount_std"]

    grouped = df.resample(WINDOW)

    feats = pd.DataFrame({
        "txn_count": grouped["amount"].count(),
        "unique_devices": grouped["device_id"].nunique(),
        "amount_mean": grouped["amount"].mean(),
        "amount_max": grouped["amount"].max(),
        "amount_z_mean": grouped["amount_z"].mean(),
        "amount_z_max": grouped["amount_z"].max(),
        "new_device_ratio": grouped["is_known_device"].apply(
            lambda s: 1 - s.mean() if len(s) > 0 else 0
        ),
        "new_geo_ratio": grouped["is_known_geo"].apply(
            lambda s: 1 - s.mean() if len(s) > 0 else 0
        ),
        "window_label": grouped["is_fraud_spike"].max(),
    })

    # drop empty windows (no transactions at all -> not meaningful)
    feats = feats[feats["txn_count"] > 0].copy()
    feats["device_reuse_rate"] = feats["txn_count"] / feats["unique_devices"]
    feats["window_label"] = feats["window_label"].fillna(0).astype(int)
    feats = feats.fillna(0)

    return feats.reset_index().rename(columns={"timestamp": "window_start"})


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
    print("\nSample feature rows (positive windows):")
    print(train_feats[train_feats.window_label == 1].head(3).to_string())


if __name__ == "__main__":
    main()
