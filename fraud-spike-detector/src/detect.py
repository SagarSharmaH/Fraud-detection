"""
Training and prediction pipeline for all fraud-spike detectors.

Trains 6 detectors on the training feature set, generates predictions on the
held-out test set, and saves everything for evaluation and visualization.

Detectors:
  1. Rule-Based Baseline     — transparent threshold rules
  2. Random Forest            — feature-combo learner
  3. XGBoost                  — gradient boosting (industry standard)
  4. Isolation Forest         — unsupervised anomaly detection
  5. Autoencoder (Neural Net) — reconstruction-error anomaly detection
  6. Ensemble (Weighted Vote) — combines all detectors

All trained models are serialized to models/ for API serving.
"""

import os
import time
import numpy as np
import pandas as pd

from models import (
    FEATURE_COLS, RuleBasedDetector, RandomForestDetector,
    XGBoostDetector, IsolationForestDetector, AutoencoderDetector,
    EnsembleDetector, build_all_detectors, HAS_XGBOOST,
)
from model_store import save_model, save_metadata


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_feats = pd.read_csv("data/features_train.csv")
    test_feats = pd.read_csv("data/features_test.csv")

    print(f"Training set: {len(train_feats)} windows ({train_feats.window_label.sum()} positive)")
    print(f"Test set:     {len(test_feats)} windows ({test_feats.window_label.sum()} positive)")
    print(f"Features:     {len(FEATURE_COLS)}")
    print()

    # ---- Build and train individual detectors ----
    detectors = build_all_detectors()

    timing = {}
    for det in detectors:
        print(f"Training {det.name}...", end=" ", flush=True)
        t0 = time.time()
        det.fit(train_feats)
        elapsed = time.time() - t0
        timing[det.name] = elapsed
        print(f"done ({elapsed:.2f}s)")

    # ---- Build ensemble (requires fitted sub-detectors) ----
    ensemble = EnsembleDetector(detectors=list(detectors))
    ensemble.fit(train_feats)
    detectors.append(ensemble)
    timing[ensemble.name] = 0.0  # ensemble "training" is just weight computation
    print(f"\nEnsemble weights: {ensemble.weights}")

    # ---- Print detector info ----
    print("\n=== Detector Details ===")
    for det in detectors:
        if isinstance(det, RuleBasedDetector):
            print(f"\n[{det.name}] Thresholds:")
            for k, v in det.thresholds.items():
                print(f"  {k}: {v:.3f}")
        elif hasattr(det, "feature_importances_") and det.feature_importances_:
            print(f"\n[{det.name}] Top feature importances:")
            sorted_imp = sorted(det.feature_importances_.items(), key=lambda x: -x[1])
            for k, v in sorted_imp[:5]:
                print(f"  {k}: {v:.3f}")

    # ---- Generate predictions on test set ----
    print("\n=== Test Set Predictions ===")
    out = test_feats.copy()

    for det in detectors:
        safe_name = det.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        preds, scores, reasons = det.predict(test_feats)

        out[f"{safe_name}_pred"] = preds
        out[f"{safe_name}_score"] = scores
        out[f"{safe_name}_reasons"] = [", ".join(r) if r else "" for r in reasons]

        print(f"  {det.name}: flagged {preds.sum()} / {len(test_feats)} windows")

    out.to_csv("data/predictions_test.csv", index=False)
    print("\nSaved predictions to data/predictions_test.csv")

    # ---- Save trained models ----
    for det in detectors:
        save_model(det, "models")
    save_metadata(detectors, timing, "models")
    print("Saved trained models to models/")


if __name__ == "__main__":
    main()
