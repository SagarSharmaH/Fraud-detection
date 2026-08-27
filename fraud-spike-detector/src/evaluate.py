"""
Evaluation harness for the fraud-spike detectors.

Reports, for BOTH detectors on the held-out test set:
  - precision, recall, F1
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
    (rough proxy: a velocity burst or amount anomaly window, if it goes
    through uncontested, represents real settled fraud loss to the merchant)
  - Cost per false positive: ~INR 150
    (proxy: cost of a manual review / customer friction if a legitimate
    window gets flagged and delayed)
  These are illustrative, not calibrated to any real merchant -- the point is
  to show the TRADEOFF exists and is quantifiable, not to claim a precise
  rupee figure.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

COST_PER_FALSE_NEGATIVE = 15000  # INR, avg loss per missed fraud-spike window
COST_PER_FALSE_POSITIVE = 150    # INR, avg cost per wrongly-flagged normal window


def evaluate_detector(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    total_cost = fp * COST_PER_FALSE_POSITIVE + fn * COST_PER_FALSE_NEGATIVE

    return {
        "detector": name,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_cost": int(fp * COST_PER_FALSE_POSITIVE),
        "false_negative_cost": int(fn * COST_PER_FALSE_NEGATIVE),
        "total_estimated_cost_inr": int(total_cost),
    }


def print_failure_cases(df: pd.DataFrame, pred_col: str, reasons_col: str, name: str, n: int = 2):
    print(f"\n--- {name}: documented failure cases (not cherry-picked wins) ---")

    fps = df[(df.window_label == 0) & (df[pred_col] == 1)]
    fns = df[(df.window_label == 1) & (df[pred_col] == 0)]

    if len(fps) > 0:
        row = fps.iloc[0]
        print(f"\n[FALSE POSITIVE example] window_start={row.window_start}")
        print(f"  txn_count={row.txn_count}, unique_devices={row.unique_devices}, "
              f"new_device_ratio={row.new_device_ratio:.2f}, amount_z_max={row.amount_z_max:.2f}")
        print(f"  Flagged because: {row[reasons_col] if row[reasons_col] else '(model score above threshold)'}")
        print("  Why this likely happened: a legitimate short-lived burst of activity "
              "(e.g. a real promo-driven traffic spike) resembles VELOCITY_BURST on "
              "count-based features alone. This is the honest cost of using volume "
              "as a signal: real flash-traffic looks statistically similar to a bot burst.")
    else:
        print("\nNo false positives in test set for this detector.")

    if len(fns) > 0:
        row = fns.iloc[0]
        print(f"\n[FALSE NEGATIVE example] window_start={row.window_start}")
        print(f"  txn_count={row.txn_count}, unique_devices={row.unique_devices}, "
              f"new_device_ratio={row.new_device_ratio:.2f}, amount_z_max={row.amount_z_max:.2f}")
        print("  Why this likely happened: a low-volume window (e.g. early minute of a "
              "ramping GEO_DEVICE_CLUSTER spike, or a small AMOUNT_ANOMALY cluster split "
              "across window boundaries) doesn't cross any single threshold and doesn't "
              "look extreme enough in aggregate for the model either. This is a real "
              "limitation of window-level (vs. entity-level, e.g. per-device) detection.")
    else:
        print("\nNo false negatives in test set for this detector.")


def main():
    df = pd.read_csv("data/predictions_test.csv")
    y_true = df["window_label"].values

    results = []
    results.append(evaluate_detector("Rule-based baseline", y_true, df["rule_pred"].values))
    results.append(evaluate_detector("ML (Random Forest)", y_true, df["ml_pred"].values))

    results_df = pd.DataFrame(results)
    print("=== Metrics on held-out test set (last 15 days, never seen during training) ===\n")
    print(results_df.to_string(index=False))

    results_df.to_csv("reports/metrics.csv", index=False)

    print_failure_cases(df, "rule_pred", "rule_reasons", "Rule-based baseline")
    print_failure_cases(df, "ml_pred", "ml_reasons", "ML (Random Forest)")

    print(f"\n\n(Cost assumptions: INR {COST_PER_FALSE_POSITIVE}/false positive, "
          f"INR {COST_PER_FALSE_NEGATIVE}/false negative -- illustrative, documented in README)")
    print("Saved metrics table to reports/metrics.csv")


if __name__ == "__main__":
    main()
