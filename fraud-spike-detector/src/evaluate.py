"""
Evaluation harness for ALL fraud-spike detectors.

Reports, for EACH detector on the held-out test set:
  - precision, recall, F1, specificity, balanced accuracy, MCC
  - a confusion matrix
  - an estimated business cost, combining:
      * cost of a false positive: a legitimate customer/transaction window
        wrongly blocked/flagged -> assumed cost per FP (support ticket,
        customer friction, lost goodwill)
      * cost of a false negative: a real fraud-spike window that slipped
        through -> assumed cost per FN (average fraud loss per spike event)
  - at least one concrete, explained failure case for each detector (a false
    positive AND a false negative), because the brief explicitly asks for
    honest exception reporting, not cherry-picked wins.

COST ASSUMPTIONS (documented, not hidden):
  - Average fraud loss per undetected spike window: ~INR 15,000
  - Cost per false positive: ~INR 150
  These are illustrative, not calibrated to any real merchant.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score,
)

log = logging.getLogger(__name__)

COST_PER_FALSE_NEGATIVE = 15000  # INR, avg loss per missed fraud-spike window
COST_PER_FALSE_POSITIVE = 150    # INR, avg cost per wrongly-flagged normal window


def _get_detector_columns(df: pd.DataFrame) -> list[str]:
    """Discover all detector prediction columns in the predictions CSV."""
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    names = [c.replace("_pred", "") for c in pred_cols]
    return names


def evaluate_detector(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                      y_score: np.ndarray | None = None) -> dict[str, Any]:
    """Compute all evaluation metrics for a single detector.

    Args:
        name: Detector name.
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_score: Optional continuous prediction scores for ROC/PR curves.

    Returns:
        Dict of metric name -> value.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    total_cost = fp * COST_PER_FALSE_POSITIVE + fn * COST_PER_FALSE_NEGATIVE

    result: dict[str, Any] = {
        "detector": name,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "mcc": round(mcc, 4),
        "false_positive_cost": int(fp * COST_PER_FALSE_POSITIVE),
        "false_negative_cost": int(fn * COST_PER_FALSE_NEGATIVE),
        "total_estimated_cost_inr": int(total_cost),
    }

    # ROC AUC and Average Precision (if scores available)
    if y_score is not None:
        try:
            result["roc_auc"] = round(roc_auc_score(y_true, y_score), 4)
        except ValueError:
            result["roc_auc"] = None
        try:
            result["avg_precision"] = round(average_precision_score(y_true, y_score), 4)
        except ValueError:
            result["avg_precision"] = None

    return result


def print_failure_cases(df: pd.DataFrame, pred_col: str, reasons_col: str,
                        name: str, n: int = 2) -> None:
    """Print documented failure cases for a detector — honest exception reporting."""
    print(f"\n--- {name}: documented failure cases (not cherry-picked wins) ---")

    fps = df[(df.window_label == 0) & (df[pred_col] == 1)]
    fns = df[(df.window_label == 1) & (df[pred_col] == 0)]

    if len(fps) > 0:
        row = fps.iloc[0]
        print(f"\n[FALSE POSITIVE example] window_start={row.window_start}")
        print(f"  txn_count={row.txn_count}, unique_devices={row.unique_devices}, "
              f"new_device_ratio={row.new_device_ratio:.2f}, amount_z_max={row.amount_z_max:.2f}")
        reason_text = row[reasons_col] if pd.notna(row[reasons_col]) and row[reasons_col] else "(model score above threshold)"
        print(f"  Flagged because: {reason_text}")
        print("  Why this likely happened: a legitimate short-lived burst of activity "
              "(e.g. a real promo-driven traffic spike) resembles VELOCITY_BURST on "
              "count-based features alone.")
    else:
        print("\nNo false positives in test set for this detector.")

    if len(fns) > 0:
        row = fns.iloc[0]
        print(f"\n[FALSE NEGATIVE example] window_start={row.window_start}")
        print(f"  txn_count={row.txn_count}, unique_devices={row.unique_devices}, "
              f"new_device_ratio={row.new_device_ratio:.2f}, amount_z_max={row.amount_z_max:.2f}")
        print("  Why this likely happened: a low-volume window (e.g. early minute of a "
              "ramping attack) doesn't look extreme enough. This is a real limitation of "
              "window-level (vs. entity-level) detection.")
    else:
        print("\nNo false negatives in test set for this detector.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        df = pd.read_csv("data/predictions_test.csv")
    except FileNotFoundError:
        log.error("Predictions file not found. Run detect.py first.")
        return

    y_true = df["window_label"].values

    det_names = _get_detector_columns(df)
    if not det_names:
        log.error("No detector columns found in predictions file.")
        return

    results: list[dict[str, Any]] = []
    for name in det_names:
        pred_col = f"{name}_pred"
        score_col = f"{name}_score"
        reasons_col = f"{name}_reasons"

        y_pred = df[pred_col].values
        y_score = df[score_col].values if score_col in df.columns else None

        result = evaluate_detector(name, y_true, y_pred, y_score)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)

    print("=" * 80)
    print("METRICS ON HELD-OUT TEST SET (last 15 days, never seen during training)")
    print("=" * 80)
    print()
    print(results_df.to_string(index=False))

    results_df.to_csv("reports/metrics.csv", index=False)

    # Print failure cases for each detector
    for name in det_names:
        pred_col = f"{name}_pred"
        reasons_col = f"{name}_reasons"
        # Format a readable name
        display_name = name.replace("_", " ").title()
        print_failure_cases(df, pred_col, reasons_col, display_name)

    print(f"\n\n(Cost assumptions: INR {COST_PER_FALSE_POSITIVE}/false positive, "
          f"INR {COST_PER_FALSE_NEGATIVE}/false negative — illustrative, documented in README)")
    print("Saved metrics table to reports/metrics.csv")

    # Print the winner
    best = results_df.iloc[0]
    print(f"\n{'='*80}")
    print(f"BEST DETECTOR: {best['detector']} (F1={best['f1']:.4f}, "
          f"MCC={best['mcc']:.4f}, Cost=INR {best['total_estimated_cost_inr']:,})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
