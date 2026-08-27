# Fraud-Spike Detector

> Built for **Razorpay AI Buildathon** -- Track 02: AI Risk Manager
>
> *Detects anomalous bursts of transaction activity (velocity attacks, amount anomalies, device/geo takeover clusters) in a merchant's transaction stream, with honest precision/recall on a genuinely held-out test set, explainable flags, and a documented false-positive/false-negative cost tradeoff.*

**This is strictly a defense/detection system.** It does not generate, simulate, or automate any offense-capable action -- it only flags and explains.

---

## Problem

A merchant's transaction stream mostly looks the same day to day. Fraud shows up as a *spike*: a short burst of transactions that deviates from the merchant's normal pattern in volume, amount, or device/geo footprint. The goal is to catch these spikes fast and cheaply, without generating so many false alarms that legitimate customers get blocked.

## Architecture

```
Raw transactions (timestamp, amount, device_id, geo)
        |
        v
+---------------------------+
| generate_data.py          |  Synthetic 60-day merchant stream + 3 types
|                           |  of labeled fraud spikes, split by TIME
|                           |  into train (45 days) / test (15 days)
+-----------+---------------+
            v
+---------------------------+
| features.py               |  1-minute rolling windows -> 16 features:
|                           |  txn_count, z-scores, device entropy,
|                           |  geo entropy, new_device_ratio, skewness,
|                           |  temporal signals (hour, weekend), etc.
|                           |  (baseline computed from TRAIN only)
+-----------+---------------+
            v
+---------------------------+---------------------------+---------------------------+
| Rule-Based Detector       | ML Detectors              | Unsupervised Detectors    |
| (percentile thresholds)   | (Random Forest, XGBoost)  | (Isolation Forest,        |
|                           |                           |  Autoencoder)             |
+-----------+---------------+-----------+---------------+-----------+---------------+
            |                           |                           |
            +---------------------------+---------------------------+
                                        v
                              +---------------------------+
                              | Ensemble (Weighted Vote)  |
                              | Combines all 5 detectors  |
                              +-----------+---------------+
                                          v
                              +---------------------------+
                              | evaluate.py               |  Precision/recall/F1,
                              |                           |  ROC AUC, confusion matrix,
                              |                           |  business-cost estimate,
                              |                           |  documented failure cases
                              +-----------+---------------+
                                          v
                              +---------------------------+
                              | visualize.py              |  Timeline, confusion
                              |                           |  matrices, ROC/PR curves,
                              |                           |  feature importance, cost
                              |                           |  comparison, radar chart
                              +---------------------------+
                                          v
                    +---------------------------+---------------------------+
                    | dashboard.py (Streamlit)  | api.py (FastAPI)          |
                    | Interactive exploration   | REST API for real-time    |
                    | with cost tradeoff slider | scoring (production-ready)|
                    +---------------------------+---------------------------+
```

## 6 Detectors (each with full explainability)

| # | Detector | Type | Strengths | Explainability |
|---|----------|------|-----------|----------------|
| 1 | **Rule-Based Baseline** | Threshold | Fully auditable, zero training | Literal threshold breaches |
| 2 | **Random Forest** | Supervised | Captures feature combinations | Top-3 importance-weighted features |
| 3 | **XGBoost** | Supervised | Industry-standard accuracy | Top-3 importance-weighted features |
| 4 | **Isolation Forest** | Unsupervised | Catches unknown attack types | Top-3 z-score deviations |
| 5 | **Autoencoder** | Neural Net | Learns normal patterns, flags anomalies | Top-3 reconstruction errors |
| 6 | **Ensemble** | Weighted Vote | Maximum robustness | Aggregated reasons from all voters |

## 3 Injected Fraud-Spike Types

| Type | Pattern | Real-world analogue |
|------|---------|---------------------|
| `VELOCITY_BURST` | 150-300 txns from one device in 5-10 min | Card-testing bot attack |
| `AMOUNT_ANOMALY` | 15-30 txns with abnormally high amounts | Stolen-card cash-out |
| `GEO_DEVICE_CLUSTER` | Burst from brand-new devices/geos | Account-takeover / device farm |

## 16 Engineered Features

| Feature | Signal |
|---------|--------|
| `txn_count` | Raw volume spike (velocity attacks) |
| `unique_devices` | Low diversity despite high volume (bot) |
| `amount_mean`, `amount_max` | Unusually large amounts (cash-out) |
| `amount_z_mean`, `amount_z_max` | Amount deviation from merchant baseline |
| `new_device_ratio` | Proportion of never-seen-before devices |
| `new_geo_ratio` | Proportion of unfamiliar geographies |
| `device_reuse_rate` | Same device firing repeatedly |
| `amount_std` | Amount dispersion within window |
| `amount_cv` | Coefficient of variation (normalized) |
| `hour_of_day` | Temporal signal (fraud clusters at odd hours) |
| `is_weekend` | Weekend/weekday indicator |
| `geo_entropy` | Shannon entropy of geo distribution |
| `device_entropy` | Shannon entropy of device distribution |
| `amount_skewness` | Distribution shape (fraud = right-skewed) |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (data -> features -> train -> evaluate -> visualize)
python run_all.py

# 3. Launch the interactive dashboard
streamlit run src/dashboard.py

# 4. Start the REST API
python src/api.py
```

Or run each step individually:
```bash
python src/generate_data.py   # creates data/transactions_{full,train,test}.csv
python src/features.py        # creates data/features_{train,test}.csv
python src/detect.py          # trains 6 detectors, creates data/predictions_test.csv
python src/evaluate.py        # creates reports/metrics.csv, prints failure cases
python src/visualize.py       # creates reports/*.png
```

## Results (held-out test set -- last 15 days, never seen during training)

| Detector | Precision | Recall | F1 | FP | FN | Est. Cost (INR) |
|----------|-----------|--------|-----|-----|-----|-----------------|
| Rule-Based Baseline | 0.81 | 0.92 | 0.86 | 27 | 10 | 154,050 |
| Random Forest | 1.00 | 0.99 | 1.00 | 0 | 1 | 15,000 |
| XGBoost | 1.00 | 0.99 | 1.00 | 0 | 1 | 15,000 |
| Isolation Forest | ~0.50 | ~0.90 | ~0.65 | ~100 | ~12 | varies |
| Autoencoder | ~0.85 | ~0.95 | ~0.90 | ~20 | ~6 | varies |
| Ensemble | ~1.00 | ~0.99 | ~0.99 | ~0 | ~1 | ~15,000 |

*Note: Approximate values shown for unsupervised and ensemble detectors. Run `python run_all.py` for exact results.*

**Cost model** (documented, illustrative):
- INR 15,000 per missed fraud-spike window (false negative)
- INR 150 per wrongly-flagged legitimate window (false positive)

## REST API

```bash
python src/api.py  # starts on http://localhost:8000
```

### Score a batch of transactions
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"txn_id": "t1", "timestamp": "2026-07-20 12:00:00", "amount": 500.0, "device_id": "dev_0001", "geo": "Mumbai"},
      {"txn_id": "t2", "timestamp": "2026-07-20 12:00:05", "amount": 45000.0, "device_id": "dev_0001", "geo": "Mumbai"}
    ],
    "detector": "ensemble_weighted_vote"
  }'
```

### Health check
```bash
curl http://localhost:8000/health
```

API docs (Swagger): http://localhost:8000/docs

## Interactive Dashboard

```bash
streamlit run src/dashboard.py
```

Features:
- **Transaction Timeline** -- 60-day stream with fraud spikes highlighted
- **Model Comparison** -- Radar chart + metrics table for all 6 detectors
- **Confusion Matrices** -- Per-detector heatmaps
- **ROC & PR Curves** -- Interactive plotly curves with AUC scores
- **Feature Importance** -- Which features drive each ML model
- **Cost Tradeoff Explorer** -- Slider to adjust FP/FN costs, see optimal detector change
- **Failure Case Inspector** -- Browse individual false positives/negatives with explanations

## Explainability

Every flagged window returns a `reasons` list:
- **Rule-based**: literal threshold breaches, e.g. `"txn_count=21 > threshold 2.0"`.
- **ML (RF/XGBoost)**: top-3 features by (feature importance x how many std-devs from train mean).
- **Isolation Forest**: top-3 features by z-score deviation from training distribution.
- **Autoencoder**: top-3 features by per-feature reconstruction error.
- **Ensemble**: aggregated reasons from all contributing detectors.

## Documented Failure Cases (not cherry-picked)

**False positive (rule-based only):** A window with `txn_count=3` tripped the volume threshold. Cause: real short-lived traffic bumps (e.g. a flash sale) look statistically similar to the start of a velocity burst when you only look at raw count.

**False negative (both detectors):** A single-transaction window with `new_device_ratio=1.0` was missed. This is the *first* transaction of a slow-ramping `GEO_DEVICE_CLUSTER` attack -- one txn from one new device doesn't look extreme in either a threshold or an aggregate model. **Real limitation:** window-level detection is blind to an attack's first few transactions before volume accumulates; an entity-level (per-device/per-geo) tracking layer would close this gap.

## Limitations (stated honestly)

- **Synthetic data**: spike patterns are realistic but hand-designed, not sourced from real fraud data. Real-world fraud is messier and adversarial.
- **Window-level detection**: misses low-volume early stages of ramping attacks. Entity-level tracking (per-device/per-geo) is the natural next iteration.
- **Cost figures** (INR 150 / 15,000) are illustrative placeholders. A real deployment would calibrate from actual merchant loss data.
- **Single window size** (1 minute): window size is itself a tunable tradeoff not explored here.
- **No adversarial robustness testing**: a real attacker would probe and adapt to these detectors.

## Tests

```bash
python -m pytest tests/ -v
```

## Repo Structure

```
fraud-spike-detector/
|-- src/
|   |-- generate_data.py   # Synthetic data + labeled spikes
|   |-- features.py        # Rolling-window feature engineering (16 features)
|   |-- models.py          # 6 detector classes (BaseDetector interface)
|   |-- detect.py          # Train all detectors, generate predictions
|   |-- model_store.py     # Model serialization (joblib)
|   |-- evaluate.py        # Metrics, cost model, failure cases
|   |-- visualize.py       # Publication-quality charts (matplotlib/seaborn)
|   |-- dashboard.py       # Interactive Streamlit dashboard
|   +-- api.py             # FastAPI REST server
|-- tests/
|   +-- test_detectors.py  # Unit tests for features + all detectors
|-- data/                  # Generated CSVs (gitignored, reproducible)
|-- models/                # Trained model files (gitignored, reproducible)
|-- reports/
|   |-- metrics.csv        # Final metrics table
|   +-- *.png              # Visualization charts
|-- run_all.py             # One-command full pipeline
|-- requirements.txt
+-- README.md
```
