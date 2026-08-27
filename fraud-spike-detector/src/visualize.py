"""
Publication-quality visualization suite for the fraud-spike detector.

Generates all report charts and saves them to reports/.
Charts are designed for both the README and the Streamlit dashboard.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score,
)

# --- Style ---
plt.rcParams.update({
    "figure.facecolor": "#0e0e12",
    "axes.facecolor": "#16161d",
    "axes.edgecolor": "#2a2a3a",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0",
    "grid.color": "#2a2a3a",
    "grid.alpha": 0.5,
    "font.size": 11,
    "font.family": "sans-serif",
})

PALETTE = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ba68c8", "#4dd0e1"]


def _get_detector_columns(df: pd.DataFrame) -> list[str]:
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    return [c.replace("_pred", "") for c in pred_cols]


def _display_name(name: str) -> str:
    return name.replace("_", " ").title()


def plot_fraud_timeline(full_df_path: str = "data/transactions_full.csv",
                        output_path: str = "reports/fraud_timeline.png"):
    """Timeline of all transactions with fraud spikes highlighted."""
    df = pd.read_csv(full_df_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Resample to hourly counts
    df_ts = df.set_index("timestamp").sort_index()
    hourly_all = df_ts.resample("1h")["amount"].count()
    hourly_fraud = df_ts[df_ts.is_fraud_spike == 1].resample("1h")["amount"].count()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(hourly_all.index, hourly_all.values, alpha=0.3, color=PALETTE[0], label="Normal Traffic")
    ax.plot(hourly_all.index, hourly_all.values, color=PALETTE[0], linewidth=0.8, alpha=0.7)

    # Highlight fraud windows
    fraud_mask = hourly_fraud.reindex(hourly_all.index, fill_value=0) > 0
    for idx in hourly_all.index[fraud_mask]:
        ax.axvline(x=idx, color="#e57373", alpha=0.4, linewidth=1.5)
    ax.bar(hourly_fraud.index, hourly_fraud.values, width=0.04, color="#e57373",
           alpha=0.9, label="Fraud-Spike Txns")

    # Train/test split line
    split_date = pd.Timestamp("2026-07-16")
    ax.axvline(x=split_date, color="#ffb74d", linestyle="--", linewidth=2, label="Train/Test Split")

    ax.set_xlabel("Date")
    ax.set_ylabel("Transactions per Hour")
    ax.set_title("60-Day Transaction Timeline — Fraud Spikes Highlighted", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_confusion_matrices(pred_df_path: str = "data/predictions_test.csv",
                            output_path: str = "reports/confusion_matrices.png"):
    """Side-by-side confusion matrix heatmaps for all detectors."""
    df = pd.read_csv(pred_df_path)
    det_names = _get_detector_columns(df)
    y_true = df["window_label"].values

    n_det = len(det_names)
    cols = min(3, n_det)
    rows = (n_det + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n_det == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(det_names):
        y_pred = df[f"{name}_pred"].values
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"],
                    annot_kws={"fontsize": 14, "fontweight": "bold"})
        axes[i].set_title(_display_name(name), fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices — Held-Out Test Set", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_roc_curves(pred_df_path: str = "data/predictions_test.csv",
                    output_path: str = "reports/roc_curves.png"):
    """ROC curves for all detectors on the same axes."""
    df = pd.read_csv(pred_df_path)
    det_names = _get_detector_columns(df)
    y_true = df["window_label"].values

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, name in enumerate(det_names):
        score_col = f"{name}_score"
        if score_col not in df.columns:
            continue
        y_score = df[score_col].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], linewidth=2,
                label=f"{_display_name(name)} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#555", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Detectors", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_pr_curves(pred_df_path: str = "data/predictions_test.csv",
                   output_path: str = "reports/pr_curves.png"):
    """Precision-Recall curves for all detectors."""
    df = pd.read_csv(pred_df_path)
    det_names = _get_detector_columns(df)
    y_true = df["window_label"].values

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, name in enumerate(det_names):
        score_col = f"{name}_score"
        if score_col not in df.columns:
            continue
        y_score = df[score_col].values
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color=PALETTE[i % len(PALETTE)], linewidth=2,
                label=f"{_display_name(name)} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Detectors", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_feature_importance(pred_df_path: str = "data/predictions_test.csv",
                            metadata_path: str = "models/metadata.json",
                            output_path: str = "reports/feature_importance.png"):
    """Feature importance comparison across ML detectors."""
    import json
    with open(metadata_path) as f:
        meta = json.load(f)

    # Collect detectors with feature importances
    importance_data = {}
    for det in meta["detectors"]:
        if "top_features" in det:
            importance_data[det["name"]] = det["top_features"]

    if not importance_data:
        print("  Skipped: no feature importance data available")
        return

    fig, axes = plt.subplots(1, len(importance_data), figsize=(6 * len(importance_data), 5))
    if len(importance_data) == 1:
        axes = [axes]

    for ax, (name, imp) in zip(axes, importance_data.items()):
        features = list(imp.keys())
        values = list(imp.values())
        colors = PALETTE[:len(features)]
        bars = ax.barh(features, values, color=colors, edgecolor="#2a2a3a")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9, color="#e0e0e0")

    fig.suptitle("Top Feature Importances — ML Detectors", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_cost_comparison(metrics_path: str = "reports/metrics.csv",
                         output_path: str = "reports/cost_comparison.png"):
    """Bar chart comparing estimated business cost across detectors."""
    df = pd.read_csv(metrics_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: F1 comparison
    colors = PALETTE[:len(df)]
    bars = ax1.barh(df["detector"], df["f1"], color=colors, edgecolor="#2a2a3a")
    ax1.set_xlabel("F1 Score")
    ax1.set_title("F1 Score Comparison", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 1.05)
    for bar, val in zip(bars, df["f1"]):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=10, color="#e0e0e0")

    # Right: Cost comparison
    x = np.arange(len(df))
    width = 0.35
    ax2.bar(x - width / 2, df["false_positive_cost"], width, label="FP Cost (₹150/ea)",
            color=PALETTE[4], edgecolor="#2a2a3a")
    ax2.bar(x + width / 2, df["false_negative_cost"], width, label="FN Cost (₹15k/ea)",
            color=PALETTE[3], edgecolor="#2a2a3a")
    ax2.set_xlabel("Detector")
    ax2.set_ylabel("Cost (INR)")
    ax2.set_title("Business Cost Breakdown", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["detector"], rotation=30, ha="right", fontsize=8)
    ax2.legend(framealpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_metrics_radar(metrics_path: str = "reports/metrics.csv",
                       output_path: str = "reports/metrics_radar.png"):
    """Radar chart comparing detectors across multiple metrics."""
    df = pd.read_csv(metrics_path)
    metrics = ["precision", "recall", "f1"]
    available_metrics = [m for m in metrics if m in df.columns]
    if "roc_auc" in df.columns:
        available_metrics.append("roc_auc")

    if len(available_metrics) < 3:
        print("  Skipped radar: not enough metrics")
        return

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] if pd.notna(row.get(m)) else 0 for m in available_metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=PALETTE[i % len(PALETTE)],
                label=row["detector"])
        ax.fill(angles, values, alpha=0.1, color=PALETTE[i % len(PALETTE)])

    ax.set_thetagrids(np.degrees(angles[:-1]), available_metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title("Multi-Metric Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.8, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    os.makedirs("reports", exist_ok=True)
    print("Generating visualizations...")

    plot_fraud_timeline()
    plot_confusion_matrices()
    plot_roc_curves()
    plot_pr_curves()
    plot_feature_importance()
    plot_cost_comparison()
    plot_metrics_radar()

    print("\nAll visualizations saved to reports/")


if __name__ == "__main__":
    main()
