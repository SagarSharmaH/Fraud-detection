"""
Model serialization and deserialization using joblib.

Provides save/load utilities so trained detectors can be persisted to disk
and loaded for API serving without re-training.
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_model(detector, model_dir: str = "models"):
    """Save a fitted detector to disk."""
    os.makedirs(model_dir, exist_ok=True)
    safe_name = detector.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(model_dir, f"{safe_name}.joblib")
    joblib.dump(detector, path)
    return path


def load_model(name: str, model_dir: str = "models"):
    """Load a fitted detector from disk."""
    path = os.path.join(model_dir, f"{name}.joblib")
    return joblib.load(path)


def load_all_models(model_dir: str = "models"):
    """Load all saved detectors from a directory."""
    detectors = []
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".joblib"):
            path = os.path.join(model_dir, fname)
            detectors.append(joblib.load(path))
    return detectors


def save_metadata(detectors, timing: dict, model_dir: str = "models"):
    """Save model metadata (names, training time, feature list) to JSON."""
    from models import FEATURE_COLS
    meta = {
        "created_at": datetime.now().isoformat(),
        "n_features": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "detectors": [],
    }
    for det in detectors:
        entry = {
            "name": det.name,
            "class": type(det).__name__,
            "training_time_seconds": round(float(timing.get(det.name, 0)), 3),
        }
        if hasattr(det, "feature_importances_") and det.feature_importances_:
            entry["top_features"] = {
                k: float(round(v, 4))
                for k, v in sorted(det.feature_importances_.items(), key=lambda x: -x[1])[:5]
            }
        if hasattr(det, "thresholds"):
            entry["thresholds"] = {k: float(round(v, 4)) for k, v in det.thresholds.items()}
        if hasattr(det, "weights"):
            entry["ensemble_weights"] = {k: float(round(v, 4)) for k, v in det.weights.items()}
        meta["detectors"].append(entry)

    path = os.path.join(model_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, cls=NumpyEncoder)
    return path
