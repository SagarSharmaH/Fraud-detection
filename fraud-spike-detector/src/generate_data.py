"""
Synthetic merchant transaction generator with injected, labeled fraud spikes.

Design:
- We simulate ONE merchant's transaction stream over 60 days.
- "Normal" traffic has daily seasonality (busier in daytime), weekly seasonality
  (weekends busier for a retail-like merchant), a stable amount distribution,
  and a realistic spread of repeat + new customers/devices/geos.
- We inject 3 distinct types of fraud spikes at random points in time, each
  labeled, so we have ground truth to evaluate against later:
    1. VELOCITY_BURST   - many transactions in a very short window (card testing /
                           bot attack pattern)
    2. AMOUNT_ANOMALY   - a cluster of transactions with abnormally high amounts
                           from otherwise normal-looking customers (stolen card
                           cash-out pattern)
    3. GEO_DEVICE_CLUSTER - a burst of transactions from a small set of new
                           devices/geos never seen before, in a short window
                           (account-takeover / device-farm pattern)

We split by TIME (not random shuffle) into train/test so the test set is a
genuinely held-out future period, which is the honest way to evaluate this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N_DAYS = 60
START = datetime(2026, 6, 1)

NORMAL_GEOS: list[str] = ["Mumbai", "Bangalore", "Delhi", "Pune", "Chennai", "Hyderabad"]
NORMAL_DEVICES_POOL_SIZE = 400  # pool of "known" recurring devices


@dataclass
class SpikeEvent:
    """Record of a single injected fraud spike event."""
    kind: str
    start: datetime
    end: datetime
    n_injected: int


def _daily_intensity(ts: pd.Timestamp) -> float:
    """Relative transaction intensity by hour-of-day and day-of-week."""
    hour_factor = 0.3 + 0.7 * np.sin((ts.hour - 6) / 24 * 2 * np.pi) ** 2
    weekend_factor = 1.3 if ts.dayofweek >= 5 else 1.0
    return float(hour_factor * weekend_factor)


def generate_normal_transactions(n_days: int = N_DAYS) -> tuple[pd.DataFrame, int]:
    """Generate normal (non-fraudulent) merchant transactions.

    Args:
        n_days: Number of days to simulate.

    Returns:
        Tuple of (DataFrame of transactions, next available txn_id counter).
    """
    if n_days <= 0:
        raise ValueError(f"n_days must be positive, got {n_days}")

    rows: list[dict] = []
    txn_id = 0
    device_pool = [f"dev_{i:04d}" for i in range(NORMAL_DEVICES_POOL_SIZE)]

    for day in range(n_days):
        date = START + timedelta(days=day)
        # base volume per day with mild random noise
        base_volume = rng.integers(180, 260)
        for _ in range(base_volume):
            # pick a random second in the day, weighted by intensity via rejection sampling
            while True:
                sec = rng.integers(0, 86400)
                ts = date + timedelta(seconds=int(sec))
                if rng.random() < _daily_intensity(pd.Timestamp(ts)):
                    break
            amount = max(50, rng.lognormal(mean=6.5, sigma=0.8))  # INR-like amounts
            device = rng.choice(device_pool)
            geo = rng.choice(NORMAL_GEOS, p=[0.28, 0.22, 0.18, 0.12, 0.11, 0.09])
            rows.append({
                "txn_id": f"t{txn_id:06d}",
                "timestamp": ts,
                "amount": round(float(amount), 2),
                "device_id": device,
                "geo": geo,
                "is_fraud_spike": 0,
                "spike_type": "none",
            })
            txn_id += 1

    df = pd.DataFrame(rows)
    log.info("Generated %d normal transactions over %d days", len(df), n_days)
    return df, txn_id


def _spike_days(n_per_period: int, n_days: int, split_day: int) -> list[int]:
    """
    Return a list of days to place spikes on, guaranteeing coverage of BOTH the
    train period (before split_day) and the test period (from split_day on).
    Without this, random placement can (and did, on first run) put zero spikes
    in the held-out test set, which would make evaluation meaningless.
    """
    if split_day <= 2 or split_day >= n_days - 2:
        raise ValueError(f"split_day must be between 3 and {n_days - 3}, got {split_day}")

    train_days = rng.choice(np.arange(2, split_day - 1), size=n_per_period, replace=False)
    test_days = rng.choice(np.arange(split_day + 1, n_days - 2), size=n_per_period, replace=False)
    return list(train_days) + list(test_days)


def inject_spikes(df: pd.DataFrame, start_txn_id: int,
                  n_days: int = N_DAYS, split_day: int = 45) -> tuple[pd.DataFrame, list[SpikeEvent]]:
    """Inject 3 kinds of labeled fraud spikes at random points in time.

    Args:
        df: Normal transaction DataFrame (used for context, not mutated).
        start_txn_id: Next available txn_id counter.
        n_days: Total number of simulation days.
        split_day: Day index for train/test split.

    Returns:
        Tuple of (DataFrame of spike transactions, list of SpikeEvent records).
    """
    rows: list[dict] = []
    txn_id = start_txn_id
    events: list[SpikeEvent] = []

    n_spikes_per_type_per_period = 4  # -> 4 in train + 4 in test, per spike type

    # 1. VELOCITY_BURST: ~150-300 txns crammed into a 5-10 min window
    for day in _spike_days(n_spikes_per_type_per_period, n_days, split_day):
        start = START + timedelta(days=int(day), seconds=int(rng.integers(0, 86000)))
        window_secs = int(rng.integers(300, 600))
        n_injected = int(rng.integers(150, 300))
        device = f"botdev_{rng.integers(0,9999):04d}"
        for i in range(n_injected):
            ts = start + timedelta(seconds=int(rng.integers(0, window_secs)))
            amount = max(50, rng.lognormal(mean=5.0, sigma=0.4))  # smaller, uniform "testing" amounts
            rows.append({
                "txn_id": f"t{txn_id:06d}",
                "timestamp": ts,
                "amount": round(float(amount), 2),
                "device_id": device,
                "geo": rng.choice(NORMAL_GEOS),
                "is_fraud_spike": 1,
                "spike_type": "VELOCITY_BURST",
            })
            txn_id += 1
        events.append(SpikeEvent("VELOCITY_BURST", start, start + timedelta(seconds=window_secs), n_injected))

    # 2. AMOUNT_ANOMALY: 15-30 txns with very high amounts in a short window
    for day in _spike_days(n_spikes_per_type_per_period, n_days, split_day):
        start = START + timedelta(days=int(day), seconds=int(rng.integers(0, 86000)))
        window_secs = int(rng.integers(600, 1800))
        n_injected = int(rng.integers(15, 30))
        for i in range(n_injected):
            ts = start + timedelta(seconds=int(rng.integers(0, window_secs)))
            amount = rng.uniform(25000, 90000)  # far above normal lognormal(6.5,0.8) range
            rows.append({
                "txn_id": f"t{txn_id:06d}",
                "timestamp": ts,
                "amount": round(float(amount), 2),
                "device_id": rng.choice([f"dev_{i:04d}" for i in range(NORMAL_DEVICES_POOL_SIZE)]),
                "geo": rng.choice(NORMAL_GEOS),
                "is_fraud_spike": 1,
                "spike_type": "AMOUNT_ANOMALY",
            })
            txn_id += 1
        events.append(SpikeEvent("AMOUNT_ANOMALY", start, start + timedelta(seconds=window_secs), n_injected))

    # 3. GEO_DEVICE_CLUSTER: burst from a small set of brand-new devices/geos
    exotic_geos = ["Lagos", "Manila", "Kyiv", "Bogota", "Jakarta"]
    for day in _spike_days(n_spikes_per_type_per_period, n_days, split_day):
        start = START + timedelta(days=int(day), seconds=int(rng.integers(0, 86000)))
        window_secs = int(rng.integers(300, 900))
        n_injected = int(rng.integers(20, 50))
        cluster_devices = [f"newdev_{rng.integers(0,9999):04d}" for _ in range(rng.integers(3, 6))]
        cluster_geo = rng.choice(exotic_geos)
        for i in range(n_injected):
            ts = start + timedelta(seconds=int(rng.integers(0, window_secs)))
            amount = max(50, rng.lognormal(mean=6.8, sigma=0.6))
            rows.append({
                "txn_id": f"t{txn_id:06d}",
                "timestamp": ts,
                "amount": round(float(amount), 2),
                "device_id": rng.choice(cluster_devices),
                "geo": cluster_geo,
                "is_fraud_spike": 1,
                "spike_type": "GEO_DEVICE_CLUSTER",
            })
            txn_id += 1
        events.append(SpikeEvent("GEO_DEVICE_CLUSTER", start, start + timedelta(seconds=window_secs), n_injected))

    spike_df = pd.DataFrame(rows)
    log.info("Injected %d spike events (%d total spike transactions)",
             len(events), len(spike_df))
    return spike_df, events


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    split_day = 45
    normal_df, next_id = generate_normal_transactions()
    spike_df, events = inject_spikes(normal_df, next_id, split_day=split_day)

    full_df = pd.concat([normal_df, spike_df], ignore_index=True)
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)

    # TIME-BASED split: first 45 days = train, last 15 days = held-out test.
    # This matters: random shuffling would leak future spike patterns into training.
    split_date = START + timedelta(days=split_day)
    train_df = full_df[full_df["timestamp"] < split_date].copy()
    test_df = full_df[full_df["timestamp"] >= split_date].copy()

    full_df.to_csv("data/transactions_full.csv", index=False)
    train_df.to_csv("data/transactions_train.csv", index=False)
    test_df.to_csv("data/transactions_test.csv", index=False)

    print(f"Total transactions: {len(full_df)}")
    print(f"  Normal: {(full_df.is_fraud_spike == 0).sum()}")
    print(f"  Fraud-spike txns: {(full_df.is_fraud_spike == 1).sum()}")
    print(f"\nSpike events injected: {len(events)}")
    for e in events:
        print(f"  [{e.kind}] {e.start} -> {e.end}  ({e.n_injected} txns)")
    print(f"\nTrain set: {len(train_df)} rows  ({train_df.timestamp.min()} to {train_df.timestamp.max()})")
    print(f"Test set:  {len(test_df)} rows  ({test_df.timestamp.min()} to {test_df.timestamp.max()})")
    print(f"\nTrain fraud-spike rate: {train_df.is_fraud_spike.mean():.4f}")
    print(f"Test fraud-spike rate:  {test_df.is_fraud_spike.mean():.4f}")


if __name__ == "__main__":
    main()
