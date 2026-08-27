# 🛡️ SentinelRisk-AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker Containerized](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CI Build](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-green.svg)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production-Grade AI Risk & Fraud-Spike Detection Engine for Payment Streams**
> 
> *SentinelRisk-AI detects anomalous transaction velocity bursts, amount anomalies, and device/geography takeover clusters in digital payment streams. Features a zero-leakage time-based split, 6 multi-paradigm detectors, explainable AI (XAI) reason codes, financial cost-tradeoff optimization, an interactive Streamlit dashboard, and a production FastAPI REST API.*

---

## 📌 Executive Summary

Modern payment processing networks require real-time risk engines that identify sudden adversarial attacks without introducing unnecessary friction for legitimate merchants. **SentinelRisk-AI** solves this by providing:

1. **Multi-Paradigm Detection Arsenal**: Combines heuristic baseline rules, supervised tree ensembles (**Random Forest**, **XGBoost**), unsupervised anomaly detectors (**Isolation Forest**, **PCA Autoencoder**), and an **F1-Weighted Ensemble**.
2. **Vectorized Feature Engine**: Computes 16 rolling-window spatial, temporal, amount distribution, and entropy metrics in native vectorized NumPy/Pandas operations ($100\times$ speedup).
3. **Explainable AI (XAI)**: Every flagged anomaly returns actionable human-readable explanations (z-score deviations, feature importance weights, reconstruction error drivers).
4. **Financial Cost-Aware Optimization**: Evaluates detectors not just on ML accuracy ($F1$, ROC-AUC), but on **actual business financial impact** by balancing False Positive friction cost ($\text{Cost}_{\text{FP}}$) vs False Negative chargeback loss ($\text{Cost}_{\text{FN}}$).

---

## 🏗️ Architecture & Data Pipeline

```
Raw Transaction Stream (Timestamp, Amount, Device ID, Geography)
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Synthetic Data Stream Engine (src/generate_data.py)          │
│ 60-Day Merchant Stream (15,086 txns) + 3 Fraud Burst Types      │
│ Split strictly by TIME: Train (45 Days) ──► Held-Out Test (15D) │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Vectorized Feature Engine (src/features.py)                  │
│ 1-Minute Resampled Windows ──► 16 Engineered Features:          │
│ Volume, Z-Scores, Geo/Device Entropy, Skewness, CV, Temporal   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Multi-Model Detection Arsenal (src/models.py)                │
│ ├─ Rule-Based Threshold Baseline  ├─ Isolation Forest Anomaly  │
│ ├─ Supervised Random Forest       ├─ Linear PCA Autoencoder     │
│ ├─ Supervised XGBoost             └─ F1-Weighted Ensemble       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Evaluation & Financial Cost Engine (src/evaluate.py)         │
│ Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix,       │
│ Business Financial Cost Breakdown & Failure Case Inspector      │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────┬──────────────────────────────┐
│ 5. Interactive Streamlit App     │ 6. FastAPI REST Engine       │
│    (src/dashboard.py)            │    (src/api.py)              │
│    Live UI on http://localhost:8501 │    Endpoint on port 8000     │
└──────────────────────────────────┴──────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Local Python Setup

```bash
# Clone repository
git clone https://github.com/SagarSharmaH/Fraud-detection.git
cd Fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run full pipeline end-to-end (24s execution)
python run_all.py

# Run test suite
python -m pytest tests/ -v
```

### 2. Launch Services

```bash
# Launch interactive Streamlit Dashboard
python -m streamlit run src/dashboard.py

# Start FastAPI REST server
python src/api.py
```

### 3. Docker Deployment (One-Command)

```bash
# Build and start both API (port 8000) and Dashboard (port 8501)
docker compose up --build
```

---

## 📊 Held-Out Test Set Performance Benchmark

Evaluated on 15 days of never-before-seen held-out transaction data ($3,158$ 1-minute windows, $122$ true positive fraud spikes):

| Detector | Paradigm | TP | FP | FN | Precision | Recall | F1 Score | ROC-AUC | Total Estimated Cost (INR)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🌲 **Random Forest** | Supervised | 115 | 0 | 7 | **1.0000** | **0.9426** | **0.9705** | **1.0000** | **₹105,000** |
| 🚀 **XGBoost** | Supervised | 115 | 2 | 7 | **0.9829** | **0.9426** | **0.9623** | **0.9999** | **₹105,300** |
| ⚖️ **Weighted Ensemble** | Meta-Voter | 115 | 13 | 7 | **0.8984** | **0.9426** | **0.9200** | **0.9997** | **₹106,950** |
| 🌲 **Isolation Forest** | Unsupervised | 113 | 22 | 9 | **0.8370** | **0.9262** | **0.8794** | **0.9926** | **₹138,300** |
| 📏 **Rule Baseline** | Heuristic | 112 | 27 | 10 | **0.8058** | **0.9180** | **0.8582** | **0.9573** | **₹154,050** |
| 🧠 **Autoencoder** | Neural Linear | 118 | 92 | 4 | **0.5619** | **0.9672** | **0.7108** | **0.9936** | **₹73,800** |

*\*Cost Model Assumptions: $\text{Cost}_{\text{FP}} = ₹150$ (manual review friction cost per false alarm); $\text{Cost}_{\text{FN}} = ₹15,000$ (average chargeback financial loss per uncaptured fraud spike).*

---

## 🔍 Injected Fraud Patterns & Feature Engine

### Attack Types
1. **`VELOCITY_BURST`**: 150--300 rapid transactions from a single device/IP in 5--10 minutes (card-testing botnet attack).
2. **`AMOUNT_ANOMALY`**: 15--30 transactions with extreme high-dollar values (stolen card cash-out burst).
3. **`GEO_DEVICE_CLUSTER`**: Rapid transaction cluster originating from brand-new device IDs and geographies (account takeover / farm attack).

### 16 Vectorized Features (`src/features.py`)
- **Volume & Velocity**: `txn_count`, `unique_devices`, `device_reuse_rate`
- **Amount Moments & Deviations**: `amount_mean`, `amount_max`, `amount_std`, `amount_cv`, `amount_skewness`, `amount_z_mean`, `amount_z_max`
- **Identity & Geographic Novelty**: `new_device_ratio`, `new_geo_ratio`, `geo_entropy`, `device_entropy`
- **Temporal Signals**: `hour_of_day`, `is_weekend`

---

## 📡 REST API Documentation

### POST `/score` — Batch Transaction Scoring

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "txn_id": "tx_1001",
        "timestamp": "2026-07-20 14:05:00",
        "amount": 120.50,
        "device_id": "dev_9981",
        "geo": "Mumbai"
      },
      {
        "txn_id": "tx_1002",
        "timestamp": "2026-07-20 14:05:12",
        "amount": 95000.00,
        "device_id": "dev_9981",
        "geo": "Mumbai"
      }
    ],
    "detector": "random_forest"
  }'
```

#### Example Response
```json
{
  "detector_used": "Random Forest",
  "n_windows": 1,
  "n_flagged": 1,
  "flagged_windows": [
    {
      "window_start": "2026-07-20 14:05:00",
      "txn_count": 2,
      "predicted_label": 1,
      "score": 0.94,
      "reasons": [
        "amount_max (importance=0.181)",
        "amount_z_max (importance=0.121)"
      ]
    }
  ],
  "scored_at": "2026-08-27T23:30:00.123456"
}
```

---

## 🧪 Testing & Verification

The test suite covers feature calculation sanity, statistical boundaries, model interfaces, threshold behavior, and FastAPI endpoint contracts:

```bash
python -m pytest tests/ -v
```

```
tests/test_api.py::test_health_endpoint PASSED                           [  6%]
tests/test_api.py::test_models_endpoint PASSED                           [ 12%]
tests/test_api.py::test_score_endpoint PASSED                            [ 18%]
tests/test_detectors.py::TestFeatures::test_historical_stats_keys PASSED [ 25%]
tests/test_detectors.py::TestFeatures::test_feature_columns_present PASSED [ 31%]
tests/test_detectors.py::TestFeatures::test_no_nans PASSED               [ 37%]
tests/test_detectors.py::TestFeatures::test_txn_count_positive PASSED    [ 43%]
tests/test_detectors.py::TestFeatures::test_window_label_binary PASSED   [ 50%]
tests/test_detectors.py::TestFeatures::test_device_reuse_rate PASSED     [ 56%]
tests/test_detectors.py::TestDetectors::test_build_all_detectors PASSED  [ 62%]
tests/test_detectors.py::TestDetectors::test_detector_interface PASSED   [ 68%]
tests/test_detectors.py::TestDetectors::test_rule_based_thresholds PASSED [ 75%]
tests/test_detectors.py::TestDetectors::test_random_forest_importances PASSED [ 81%]
tests/test_detectors.py::TestDetectors::test_reasons_only_for_flagged PASSED [ 87%]
tests/test_detectors.py::TestDetectors::test_ensemble_weights_sum_to_one PASSED [ 93%]
tests/test_detectors.py::TestDetectors::test_autoencoder_threshold PASSED [100%]

====================== 16 passed in 7.66s ======================
```

---

## 💼 LinkedIn Showcase Post Template

Feel free to share your project on LinkedIn using this ready-to-use template:

```markdown
🚀 Excited to share my latest project: SentinelRisk-AI — Production-Grade AI Risk & Transaction Fraud Engine! 🛡️

Detecting sudden fraud spikes (velocity attacks, card-testing botnets, and account takeover clusters) in real-time digital payment streams requires high precision, zero data leakage, and clear explainability.

I built SentinelRisk-AI to solve this with an end-to-end Machine Learning & Risk System:

💡 Key Features:
🔹 Vectorized Feature Engine: 16 rolling-window statistical, entropy, and temporal features (100x speedup).
🔹 Multi-Model Detector Arsenal: Heuristic Rule-Based, Supervised (Random Forest, XGBoost), Unsupervised (Isolation Forest, PCA Autoencoder), and F1-Weighted Ensembles.
🔹 Explainable AI (XAI): Returns human-readable reason codes for every flagged transaction window.
🔹 Financial Cost Optimization: Evaluates models based on real-world False Positive review costs vs. False Negative chargeback losses.
🔹 Interactive Dashboard & REST API: Built with Streamlit, FastAPI, Docker, and GitHub Actions CI/CD.

📈 Results on 15-day held-out test set:
- Random Forest: 100% Precision, 94.26% Recall (F1 = 0.9705)
- XGBoost: 98.29% Precision, 94.26% Recall (F1 = 0.9623)

🔗 Check out the GitHub Repository: https://github.com/SagarSharmaH/Fraud-detection

#MachineLearning #Python #AI #FraudDetection #FinTech #DataScience #FastAPI #Streamlit #Docker #OpenSource
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
