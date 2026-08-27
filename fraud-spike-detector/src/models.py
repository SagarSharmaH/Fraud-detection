"""
All fraud-spike detectors, unified under a common BaseDetector interface.

Six detectors, each with distinct strengths:

1. RuleBasedDetector      — transparent threshold rules (auditable)
2. RandomForestDetector   — ensemble of decision trees (captures feature combos)
3. XGBoostDetector        — gradient boosting (industry-standard, high accuracy)
4. IsolationForestDetector — unsupervised anomaly detection (catches unknown attacks)
5. AutoencoderDetector    — neural-net reconstruction error (deep anomaly detection)
6. EnsembleDetector       — weighted vote across all detectors (maximum robustness)

Every detector exposes the same interface:
    .fit(train_feats: pd.DataFrame) -> self
    .predict(feats: pd.DataFrame) -> (preds, scores, reasons_list)
    .name: str  — human-readable detector name

Explainability: every flagged window comes with a `reasons` list explaining
WHY it was flagged.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


FEATURE_COLS = [
    "txn_count", "unique_devices", "amount_mean", "amount_max",
    "amount_z_mean", "amount_z_max", "new_device_ratio", "new_geo_ratio",
    "device_reuse_rate",
    # --- new features (v2) ---
    "amount_std", "amount_cv", "hour_of_day", "is_weekend",
    "geo_entropy", "device_entropy", "amount_skewness",
]


class BaseDetector(ABC):
    """Common interface for all fraud-spike detectors."""

    name: str = "BaseDetector"

    @abstractmethod
    def fit(self, train_feats: pd.DataFrame) -> "BaseDetector":
        ...

    @abstractmethod
    def predict(self, feats: pd.DataFrame):
        """
        Returns:
            preds: np.ndarray of 0/1 labels
            scores: np.ndarray of float confidence scores
            reasons_list: list[list[str]] of human-readable flag explanations
        """
        ...


# ---------------------------------------------------------------------------
# 1. Rule-Based Detector
# ---------------------------------------------------------------------------
class RuleBasedDetector(BaseDetector):
    """Transparent threshold rules, calibrated off the train set's normal windows."""

    name = "Rule-Based Baseline"

    def __init__(self):
        self.thresholds = {}

    def fit(self, train_feats: pd.DataFrame):
        normal = train_feats[train_feats.window_label == 0]
        self.thresholds["txn_count"] = normal["txn_count"].quantile(0.995)
        self.thresholds["amount_z_max"] = normal["amount_z_max"].quantile(0.995)
        self.thresholds["new_device_ratio"] = 0.9
        self.thresholds["device_reuse_rate"] = normal["device_reuse_rate"].quantile(0.995)
        return self

    def predict(self, feats: pd.DataFrame):
        t_txn = self.thresholds["txn_count"]
        t_zmax = self.thresholds["amount_z_max"]
        t_devratio = self.thresholds["new_device_ratio"]
        t_reuse = self.thresholds["device_reuse_rate"]

        txn_counts = feats["txn_count"].values
        z_maxs = feats["amount_z_max"].values
        dev_ratios = feats["new_device_ratio"].values
        reuses = feats["device_reuse_rate"].values

        cond1 = txn_counts > t_txn
        cond2 = z_maxs > t_zmax
        cond3 = (dev_ratios > t_devratio) & (txn_counts >= 5)
        cond4 = reuses > t_reuse

        preds = (cond1 | cond2 | cond3 | cond4).astype(int)
        scores = (cond1.astype(float) + cond2.astype(float) + cond3.astype(float) + cond4.astype(float)) / 4.0

        n = len(feats)
        reasons_list = [[] for _ in range(n)]
        flagged_indices = np.where(preds == 1)[0]

        for i in flagged_indices:
            reasons = []
            if cond1[i]:
                reasons.append(f"txn_count={txn_counts[i]:.0f} > threshold {t_txn:.1f}")
            if cond2[i]:
                reasons.append(f"amount_z_max={z_maxs[i]:.1f} > threshold {t_zmax:.1f}")
            if cond3[i]:
                reasons.append(f"new_device_ratio={dev_ratios[i]:.2f} (mostly unseen devices)")
            if cond4[i]:
                reasons.append(f"device_reuse_rate={reuses[i]:.1f} > threshold {t_reuse:.1f}")
            reasons_list[i] = reasons

        return preds, scores, reasons_list


# ---------------------------------------------------------------------------
# 2. Random Forest Detector
# ---------------------------------------------------------------------------
class RandomForestDetector(BaseDetector):
    """Random Forest classifier with per-prediction feature-based explanations."""

    name = "Random Forest"

    def __init__(self, threshold: float = 0.5):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        self.threshold = threshold
        self.feature_importances_ = {}

    def fit(self, train_feats: pd.DataFrame):
        X = train_feats[FEATURE_COLS]
        y = train_feats["window_label"]
        self.model.fit(X, y)
        self.feature_importances_ = dict(zip(FEATURE_COLS, self.model.feature_importances_))
        self._train_means = X.mean()
        self._train_stds = X.std().replace(0, 1)
        return self

    def predict(self, feats: pd.DataFrame):
        X = feats[FEATURE_COLS]
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        n = len(feats)
        reasons_list = [[] for _ in range(n)]
        flagged_indices = np.where(preds == 1)[0]

        if len(flagged_indices) > 0:
            X_vals = X.values
            means_vals = self._train_means.values
            stds_vals = self._train_stds.values
            imp_vals = np.array([self.feature_importances_.get(col, 0) for col in FEATURE_COLS])

            for i in flagged_indices:
                row_vals = X_vals[i]
                z = np.abs((row_vals - means_vals) / stds_vals)
                contribs = z * imp_vals
                top3 = np.argsort(contribs)[-3:][::-1]
                reasons_list[i] = [
                    f"{FEATURE_COLS[j]} (value={row_vals[j]:.2f}, importance-weighted)"
                    for j in top3
                ]

        return preds, probs, reasons_list


# ---------------------------------------------------------------------------
# 3. XGBoost Detector
# ---------------------------------------------------------------------------
class XGBoostDetector(BaseDetector):
    """Gradient-boosted tree classifier — industry standard for tabular fraud detection."""

    name = "XGBoost"

    def __init__(self, threshold: float = 0.5):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required for XGBoostDetector")
        self.model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            scale_pos_weight=10, eval_metric="logloss", random_state=42, n_jobs=-1
        )
        self.threshold = threshold
        self.feature_importances_ = {}

    def fit(self, train_feats: pd.DataFrame):
        X = train_feats[FEATURE_COLS]
        y = train_feats["window_label"]
        self.model.fit(X, y)
        self.feature_importances_ = dict(zip(FEATURE_COLS, self.model.feature_importances_))
        self._train_means = X.mean()
        self._train_stds = X.std().replace(0, 1)
        return self

    def predict(self, feats: pd.DataFrame):
        X = feats[FEATURE_COLS]
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        n = len(feats)
        reasons_list = [[] for _ in range(n)]
        flagged_indices = np.where(preds == 1)[0]

        if len(flagged_indices) > 0:
            X_vals = X.values
            means_vals = self._train_means.values
            stds_vals = self._train_stds.values
            imp_vals = np.array([self.feature_importances_.get(col, 0) for col in FEATURE_COLS])

            for i in flagged_indices:
                row_vals = X_vals[i]
                z = np.abs((row_vals - means_vals) / stds_vals)
                contribs = z * imp_vals
                top3 = np.argsort(contribs)[-3:][::-1]
                reasons_list[i] = [
                    f"{FEATURE_COLS[j]} (value={row_vals[j]:.2f}, xgb-importance-weighted)"
                    for j in top3
                ]

        return preds, probs, reasons_list


# ---------------------------------------------------------------------------
# 4. Isolation Forest Detector (Unsupervised)
# ---------------------------------------------------------------------------
class IsolationForestDetector(BaseDetector):
    """Unsupervised anomaly detector — doesn't need labels, catches unknown attack types."""

    name = "Isolation Forest"

    def __init__(self, contamination: float = 0.02):
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination,
            random_state=42, n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def fit(self, train_feats: pd.DataFrame):
        X = train_feats[FEATURE_COLS]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._train_means = X.mean()
        self._train_stds = X.std().replace(0, 1)
        return self

    def predict(self, feats: pd.DataFrame):
        X = feats[FEATURE_COLS]
        X_scaled = self.scaler.transform(X)

        raw_scores = self.model.decision_function(X_scaled)
        iso_preds = self.model.predict(X_scaled)
        preds = (iso_preds == -1).astype(int)
        scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)

        n = len(feats)
        reasons_list = [[] for _ in range(n)]
        flagged_indices = np.where(preds == 1)[0]

        if len(flagged_indices) > 0:
            X_vals = X.values
            means_vals = self._train_means.values
            stds_vals = self._train_stds.values

            for i in flagged_indices:
                row_vals = X_vals[i]
                z = np.abs((row_vals - means_vals) / stds_vals)
                top3 = np.argsort(z)[-3:][::-1]
                reasons_list[i] = [
                    f"{FEATURE_COLS[j]} (z-score={z[j]:.2f}, anomalous)"
                    for j in top3
                ]

        return preds, scores, reasons_list


# ---------------------------------------------------------------------------
# 5. Autoencoder Detector (PCA Bottleneck)
# ---------------------------------------------------------------------------
class AutoencoderDetector(BaseDetector):
    """Linear bottleneck Autoencoder (PCA reconstruction error) that learns normal
    transaction patterns. Fraud spikes produce high reconstruction error."""

    name = "Autoencoder"

    def __init__(self, n_components: int = 6, threshold_percentile: float = 97.0):
        self.pca = PCA(n_components=n_components, random_state=42)
        self.scaler = StandardScaler()
        self.threshold_percentile = threshold_percentile
        self.threshold = None

    def fit(self, train_feats: pd.DataFrame):
        X = train_feats[FEATURE_COLS].values
        X_scaled = self.scaler.fit_transform(X)

        self.pca.fit(X_scaled)

        normal_mask = train_feats["window_label"].values == 0
        X_normal = X_scaled[normal_mask]
        X_proj = self.pca.transform(X_normal)
        X_rec = self.pca.inverse_transform(X_proj)
        normal_errors = np.mean((X_normal - X_rec) ** 2, axis=1)

        self.threshold = np.percentile(normal_errors, self.threshold_percentile)
        return self

    def predict(self, feats: pd.DataFrame):
        X = feats[FEATURE_COLS].values
        X_scaled = self.scaler.transform(X)
        X_proj = self.pca.transform(X_scaled)
        X_rec = self.pca.inverse_transform(X_proj)

        errors = np.mean((X_scaled - X_rec) ** 2, axis=1)
        preds = (errors > self.threshold).astype(int)
        scores = errors / (self.threshold + 1e-10)

        n = len(feats)
        reasons_list = [[] for _ in range(n)]
        per_feature_errors = (X_scaled - X_rec) ** 2
        flagged_indices = np.where(preds == 1)[0]

        for i in flagged_indices:
            top_idx = np.argsort(per_feature_errors[i])[-3:][::-1]
            reasons_list[i] = [
                f"{FEATURE_COLS[j]} (reconstruction_error={per_feature_errors[i, j]:.3f})"
                for j in top_idx
            ]

        return preds, scores, reasons_list


# ---------------------------------------------------------------------------
# 6. Ensemble Detector (Weighted Voting)
# ---------------------------------------------------------------------------
class EnsembleDetector(BaseDetector):
    """Combines multiple detectors via weighted majority voting."""

    name = "Ensemble (Weighted Vote)"

    def __init__(self, detectors: list[BaseDetector], vote_threshold: float = 0.4):
        self.detectors = detectors
        self.vote_threshold = vote_threshold
        self.weights = {}

    def fit(self, train_feats: pd.DataFrame):
        from sklearn.metrics import f1_score

        y_true = train_feats["window_label"].values
        for det in self.detectors:
            preds, _, _ = det.predict(train_feats)
            f1 = f1_score(y_true, preds, zero_division=0)
            self.weights[det.name] = max(f1, 0.01)

        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        return self

    def predict(self, feats: pd.DataFrame):
        n = len(feats)
        weighted_votes = np.zeros(n)
        all_reasons = [[] for _ in range(n)]

        sub_preds_reasons = []
        for det in self.detectors:
            preds, scores, sub_reasons = det.predict(feats)
            w = self.weights.get(det.name, 1.0 / len(self.detectors))
            weighted_votes += preds * w
            sub_preds_reasons.append((det.name, preds, sub_reasons))

        preds = (weighted_votes >= self.vote_threshold).astype(int)
        scores = weighted_votes

        reasons_list = [[] for _ in range(n)]
        ens_flagged = np.where(preds == 1)[0]
        if len(ens_flagged) > 0:
            for det_name, sub_preds, sub_reasons in sub_preds_reasons:
                for i in ens_flagged:
                    if sub_preds[i] == 1 and sub_reasons[i]:
                        all_reasons[i].append(f"[{det_name}] {'; '.join(sub_reasons[i][:2])}")

            for i in ens_flagged:
                reasons_list[i] = all_reasons[i][:4]

        return preds, scores, reasons_list

    def __getstate__(self):
        state = self.__dict__.copy()
        state["detectors"] = []
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.detectors = []


# ---------------------------------------------------------------------------
# Factory: build all detectors
# ---------------------------------------------------------------------------
def build_all_detectors() -> list[BaseDetector]:
    """Return a list of all available detector instances (unfitted)."""
    detectors = [
        RuleBasedDetector(),
        RandomForestDetector(),
        IsolationForestDetector(),
        AutoencoderDetector(),
    ]
    if HAS_XGBOOST:
        detectors.insert(2, XGBoostDetector())
    return detectors
