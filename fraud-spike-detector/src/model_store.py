"""
Model serialization and deserialization using joblib.

Provides save/load utilities so trained detectors can be persisted to disk
and loaded for API serving without re-training.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime

import joblib
import numpy as np

log = logging.getLogger(__name__)

# Model store version — bump when serialization format changes
MODEL_STORE_VERSION = "2.0.0"


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj: object) -> object:
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_model(detector: object, model_dir: str = "models") -> str:
    """Save a fitted detector to disk.

    Args:
        detector: A fitted BaseDetector instance.
        model_dir: Directory to save models to.

    Returns:
        Path to the saved model file.
    """
    os.makedirs(model_dir, exist_ok=True)
    safe_name = detector.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(model_dir, f"{safe_name}.joblib")
    try:
        joblib.dump(detector, path)
        log.info("Saved model '%s' to %s", detector.name, path)
    except Exception as e:
        log.error("Failed to save model '%s': %s", detector.name, e)
        raise
    return path


def load_model(name: str, model_dir: str = "models") -> object:
    """Load a fitted detector from disk.

    Args:
        name: Safe name of the model (lowercase, underscored).
        model_dir: Directory to load models from.

    Returns:
        The loaded detector instance.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
        RuntimeError: If the model file is corrupted.
    """
    path = os.path.join(model_dir, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        model = joblib.load(path)
        log.info("Loaded model '%s' from %s", getattr(model, 'name', name), path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}") from e


def load_all_models(model_dir: str = "models") -> list:
    """Load all saved detectors from a directory.

    Args:
        model_dir: Directory containing .joblib model files.

    Returns:
        List of loaded detector instances. Corrupted files are skipped
        with a warning.
    """
    detectors = []
    if not os.path.isdir(model_dir):
        log.warning("Model directory does not exist: %s", model_dir)
        return detectors

    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".joblib"):
            path = os.path.join(model_dir, fname)
            try:
                det = joblib.load(path)
                detectors.append(det)
                log.info("Loaded detector: %s", getattr(det, 'name', fname))
            except Exception as e:
                log.warning("Skipping corrupted model file %s: %s", fname, e)

    log.info("Loaded %d detectors from %s", len(detectors), model_dir)
    return detectors


def save_metadata(detectors: list, timing: dict[str, float], model_dir: str = "models") -> str:
    """Save model metadata (names, training time, feature list, version) to JSON.

    Args:
        detectors: List of fitted detector instances.
        timing: Dict mapping detector name -> training time in seconds.
        model_dir: Directory to save metadata to.

    Returns:
        Path to the saved metadata file.
    """
    from models import FEATURE_COLS

    meta = {
        "version": MODEL_STORE_VERSION,
        "created_at": datetime.now().isoformat(),
        "n_features": len(FEATURE_COLS),
        "feature_columns": list(FEATURE_COLS),
        "detectors": [],
    }
    for det in detectors:
        entry: dict = {
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
    log.info("Saved metadata for %d detectors to %s", len(detectors), path)
    return path
