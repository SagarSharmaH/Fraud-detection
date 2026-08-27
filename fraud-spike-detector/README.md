# Fraud-Spike Detector

Built for Razorpay's AI Buildathon — **Track 02: AI Risk Manager**, direction: *Fraud-spike detector*.

Detects anomalous bursts of transaction activity for a merchant (velocity attacks, amount
anomalies, device/geo takeover clusters), with honest precision/recall on a genuinely
held-out test set, explainable flags, and a documented false-positive/false-negative cost tradeoff.

**This is strictly a defense/detection system.** It does not generate, simulate, or automate
any offense-capable action — it only flags and explains.

---

## Problem

A merchant's transaction stream mostly looks the same day to day. Fraud shows up as a
*spike*: a short burst of transactions that deviates from the merchant's normal pattern in
volume, amount, or device/geo footprint. The goal is to catch these spikes fast and cheaply,
without generating so many false alarms that legitimate customers get blocked.

## Architecture

```
Raw transactions (timestamp, amount, device_id, geo)
        │
        ▼
┌───────────────────────┐
│ generate_data.py       │  synthetic 60-day merchant stream + 3 types of
│                        │  labeled fraud spikes, split by TIME (not random)
│                        │  into train (45 days) / test (15 days, held-out)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ features.py            │  1-minute rolling windows → txn_count,
│                        │  unique_devices, amount z-scores, new_device_ratio,
│                        │  new_geo_ratio, device_reuse_rate
│                        │  (historical baseline computed from TRAIN only,
│                        │   to avoid leaking test-period info into features)
└───────────┬───────────┘
            ▼
┌───────────────────────┐        ┌───────────────────────┐
│ RuleBasedDetector      │        │ MLDetector             │
│ (transparent            │        │ (Random Forest,        │
│  percentile thresholds) │       │  importance-weighted    │
│                        │        │  per-flag explanations) │
└───────────┬───────────┘        └───────────┬───────────┘
            └───────────────┬────────────────┘
                             ▼
                  ┌───────────────────────┐
                  │ evaluate.py            │  precision/recall/F1,
                  │                        │  confusion matrix, business-cost
                  │                        │  estimate (FP cost vs FN cost),
                  │                        │  documented failure cases
                  └───────────────────────┘
```

## The 3 injected fraud-spike types

| Type | Pattern | Real-world analogue |
|---|---|---|
| `VELOCITY_BURST` | 150–300 txns from one device in 5–10 min | Card-testing bot attack |
| `AMOUNT_ANOMALY` | 15–30 txns with abnormally high amounts | Stolen-card cash-out |
| `GEO_DEVICE_CLUSTER` | Burst from a handful of brand-new devices/geos | Account-takeover / device farm |

## How to run

```bash
pip install -r requirements.txt

python3 src/generate_data.py   # creates data/transactions_{full,train,test}.csv
python3 src/features.py        # creates data/features_{train,test}.csv
python3 src/detect.py          # creates data/predictions_test.csv
python3 src/evaluate.py        # creates reports/metrics.csv, prints failure cases
```

## Results (held-out test set, last 15 days — never seen during training)

| Detector | Precision | Recall | F1 | Est. cost (INR) |
|---|---|---|---|---|
| Rule-based baseline | 0.81 | 0.92 | 0.86 | 154,050 |
| ML (Random Forest) | 1.00 | 0.94 | 0.97 | 105,000 |

**Cost model** (documented, illustrative — not calibrated to a real merchant):
INR 15,000 per missed fraud-spike window (false negative), INR 150 per wrongly-flagged
legitimate window (false positive, e.g. manual review friction).

The ML detector wins on both axes here because it can learn *combinations* of features
(e.g. "moderate volume + high new-device ratio"), whereas the rule-based detector can only
threshold one feature at a time. This is the expected tradeoff: rules are cheaper to audit
and explain in a compliance review, ML costs less in false positives but requires trusting
a model's importance-weighted explanation instead of a hand-written threshold.

## Explainability

Every flagged window returns a `reasons` list:
- **Rule-based**: literal threshold breaches, e.g. `"txn_count=21 > threshold 2.0"`.
- **ML**: top-3 features by (feature importance × how many std-devs the value is from the
  training mean) — a lightweight, easy-to-explain stand-in for SHAP.

## Documented failure cases (not cherry-picked)

**False positive (rule-based only):** a window with `txn_count=3` tripped the volume
threshold. Likely cause: real short-lived traffic bumps (e.g. a flash sale) look
statistically similar to the start of a velocity burst when you only look at raw count.

**False negative (both detectors):** a single-transaction window with `new_device_ratio=1.0`
was missed by both. This is the *first* transaction of a slow-ramping `GEO_DEVICE_CLUSTER`
attack — one txn from one new device doesn't look extreme in either a threshold or an
aggregate model. **Real limitation:** window-level detection is blind to an attack's first
few transactions before volume accumulates; an entity-level (per-device/per-geo) tracking
layer would likely close this gap, and is the natural next iteration.

## Limitations (stated honestly)

- Synthetic data: spike patterns are realistic but hand-designed, not sourced from real
  fraud data — real-world fraud is messier and adversarial (attackers adapt to detectors).
- Window-level (not entity-level) detection: as shown above, this misses low-volume early
  stages of a ramping attack.
- Cost figures (₹150 / ₹15,000) are illustrative placeholders to demonstrate the tradeoff
  exists and is measurable — a real deployment would calibrate these from actual merchant
  loss and support-ticket data.
- Only 1-minute windows were tested; window size is itself a tunable tradeoff (smaller
  windows catch bursts faster but are noisier) not explored here due to time constraints.

## Repo structure

```
fraud-spike-detector/
├── src/
│   ├── generate_data.py   # synthetic data + labeled spikes
│   ├── features.py        # rolling-window feature engineering
│   ├── detect.py          # rule-based + ML detectors
│   └── evaluate.py        # metrics, cost model, failure cases
├── data/                  # generated CSVs (gitignored except a sample)
├── reports/
│   └── metrics.csv        # final metrics table
└── requirements.txt
```
