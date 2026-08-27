"""
Two fraud-spike detectors, built to be directly compared:

1. RuleBasedDetector
   - Simple, fully transparent thresholds on the engineered features.
   - No training needed; thresholds are picked by inspecting the train set's
     normal-window distribution (e.g. 99th percentile of txn_count).
   - Pro: trivially explainable, zero training cost, easy to reason about in
     an audit.
   - Con: brittle — a fraud pattern that doesn't cross any single threshold
     slips through, and thresholds don't adapt to combinations of features.

2. MLDetector (Random Forest)
   - Learns from combinations of features (e.g. "medium txn_count BUT high
     new_device_ratio" -> looks like GEO_DEVICE_CLUSTER even if txn_count
     alone isn't extreme).
   - Pro: catches subtler / combined signals, generally higher recall at the
     same false-positive budget.
   - Con: less transparent by default -- we recover explainability via
     feature importances / per-prediction feature contributions, not via a
     hand-written rule.

Both detectors output, for every flagged window: the score, the predicted
label, AND a `reasons` list of which specific features drove the flag. This
satisfies the "every flag must be explainable" requirement in the brief.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLS = [
    "txn_count", "unique_devices", "amount_mean", "amount_max",
    "amount_z_mean", "amount_z_max", "new_device_ratio", "new_geo_ratio",
    "device_reuse_rate",
]


class RuleBasedDetector:
    """Transparent threshold rules, calibrated off the train set's normal windows."""

    def __init__(self):
        self.thresholds = {}

    def fit(self, train_feats: pd.DataFrame):
        normal = train_feats[train_feats.window_label == 0]
        # 99.5th percentile of normal windows -> anything beyond this is "unusual"
        self.thresholds["txn_count"] = normal["txn_count"].quantile(0.995)
        self.thresholds["amount_z_max"] = normal["amount_z_max"].quantile(0.995)
        self.thresholds["new_device_ratio"] = 0.9  # near-all-new-devices in a window
        self.thresholds["device_reuse_rate"] = normal["device_reuse_rate"].quantile(0.995)
        return self

    def predict(self, feats: pd.DataFrame):
        reasons_list = []
        preds = []
        scores = []
        for _, row in feats.iterrows():
            reasons = []
            if row["txn_count"] > self.thresholds["txn_count"]:
                reasons.append(f"txn_count={row['txn_count']:.0f} > threshold {self.thresholds['txn_count']:.1f}")
            if row["amount_z_max"] > self.thresholds["amount_z_max"]:
                reasons.append(f"amount_z_max={row['amount_z_max']:.1f} > threshold {self.thresholds['amount_z_max']:.1f}")
            if row["new_device_ratio"] > self.thresholds["new_device_ratio"] and row["txn_count"] >= 5:
                reasons.append(f"new_device_ratio={row['new_device_ratio']:.2f} (mostly unseen devices)")
            if row["device_reuse_rate"] > self.thresholds["device_reuse_rate"]:
                reasons.append(f"device_reuse_rate={row['device_reuse_rate']:.1f} > threshold {self.thresholds['device_reuse_rate']:.1f}")

            preds.append(1 if len(reasons) > 0 else 0)
            scores.append(len(reasons) / 4.0)  # crude confidence: fraction of rules tripped
            reasons_list.append(reasons)
        return np.array(preds), np.array(scores), reasons_list


class MLDetector:
    """Random Forest classifier with per-prediction feature-based explanations."""

    def __init__(self, threshold: float = 0.5):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            class_weight="balanced", random_state=42,
        )
        self.threshold = threshold

    def fit(self, train_feats: pd.DataFrame):
        X = train_feats[FEATURE_COLS]
        y = train_feats["window_label"]
        self.model.fit(X, y)
        self.feature_importances_ = dict(zip(FEATURE_COLS, self.model.feature_importances_))
        return self

    def predict(self, feats: pd.DataFrame):
        X = feats[FEATURE_COLS]
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        # Explainability: for each flagged row, report which of its features
        # are the most extreme relative to the training feature means/stds
        # weighted by that feature's global importance. This is a lightweight
        # stand-in for SHAP that's easy to explain in a panel interview.
        reasons_list = []
        means = self.model.feature_names_in_ if hasattr(self.model, "feature_names_in_") else FEATURE_COLS
        train_means = X.mean()
        train_stds = X.std().replace(0, 1)
        for i, (_, row) in enumerate(feats.iterrows()):
            if preds[i] == 0:
                reasons_list.append([])
                continue
            contributions = {}
            for col in FEATURE_COLS:
                z = (row[col] - train_means[col]) / train_stds[col]
                contributions[col] = abs(z) * self.feature_importances_[col]
            top_features = sorted(contributions.items(), key=lambda x: -x[1])[:3]
            reasons = [f"{col} (value={row[col]:.2f}, unusual & importance-weighted)" for col, _ in top_features]
            reasons_list.append(reasons)

        return preds, probs, reasons_list


def main():
    train_feats = pd.read_csv("data/features_train.csv")
    test_feats = pd.read_csv("data/features_test.csv")

    rule_detector = RuleBasedDetector().fit(train_feats)
    ml_detector = MLDetector().fit(train_feats)

    print("=== Rule-based thresholds (learned from train, normal windows only) ===")
    for k, v in rule_detector.thresholds.items():
        print(f"  {k}: {v:.3f}")

    print("\n=== ML feature importances ===")
    for k, v in sorted(ml_detector.feature_importances_.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")

    # quick sanity check on test set
    rule_preds, rule_scores, rule_reasons = rule_detector.predict(test_feats)
    ml_preds, ml_scores, ml_reasons = ml_detector.predict(test_feats)

    print(f"\nRule-based flagged {rule_preds.sum()} / {len(test_feats)} test windows")
    print(f"ML flagged {ml_preds.sum()} / {len(test_feats)} test windows")

    # save predictions for the evaluation step
    out = test_feats.copy()
    out["rule_pred"] = rule_preds
    out["rule_score"] = rule_scores
    out["rule_reasons"] = [", ".join(r) if r else "" for r in rule_reasons]
    out["ml_pred"] = ml_preds
    out["ml_score"] = ml_scores
    out["ml_reasons"] = [", ".join(r) if r else "" for r in ml_reasons]
    out.to_csv("data/predictions_test.csv", index=False)
    print("\nSaved predictions to data/predictions_test.csv")


if __name__ == "__main__":
    main()
