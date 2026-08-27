"""
Interactive Streamlit dashboard for the Fraud-Spike Detector.

Launch:  streamlit run src/dashboard.py

Sections:
  1. Transaction Timeline         — 60-day stream with fraud highlighted
  2. Model Comparison             — Metrics table + radar chart
  3. Confusion Matrices           — Per-detector heatmaps
  4. ROC & PR Curves              — Interactive plotly curves
  5. Feature Importance           — Which features drive each model
  6. Cost Tradeoff Explorer       — Interactive FP/FN cost slider
  7. Failure Case Inspector       — Browse individual FP/FN examples
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, precision_score, recall_score, f1_score,
)

# --- Page config ---
st.set_page_config(
    page_title="SentinelRisk-AI — Transaction Risk Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    h1 { color: #4fc3f7; }
    h2 { color: #81c784; }
    .stMetric > div { background: #1a1a2e; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ba68c8", "#4dd0e1"]


def _get_detector_columns(df):
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    return [c.replace("_pred", "") for c in pred_cols]


def _display_name(name):
    return name.replace("_", " ").title()


@st.cache_data
def load_data():
    data = {}
    for name, path in [
        ("full", "data/transactions_full.csv"),
        ("train", "data/transactions_train.csv"),
        ("test", "data/transactions_test.csv"),
        ("predictions", "data/predictions_test.csv"),
        ("features_train", "data/features_train.csv"),
        ("features_test", "data/features_test.csv"),
        ("metrics", "reports/metrics.csv"),
    ]:
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
    if os.path.exists("models/metadata.json"):
        with open("models/metadata.json") as f:
            data["metadata"] = json.load(f)
    return data


def main():
    st.title("🛡️ SentinelRisk-AI Engine")
    st.caption("Enterprise AI Risk Manager — Multi-Model Transaction Spike & Fraud Detection Platform")

    data = load_data()

    if "predictions" not in data:
        st.error("Run the pipeline first: `python run_all.py`")
        return

    # --- Sidebar ---
    st.sidebar.header("Navigation")
    section = st.sidebar.radio("Go to", [
        "📊 Overview",
        "📈 Transaction Timeline",
        "🏆 Model Comparison",
        "🔲 Confusion Matrices",
        "📉 ROC & PR Curves",
        "🎯 Feature Importance",
        "💰 Cost Tradeoff Explorer",
        "🔍 Failure Case Inspector",
    ])

    pred_df = data["predictions"]
    det_names = _get_detector_columns(pred_df)
    y_true = pred_df["window_label"].values

    # =====================================================
    # 1. OVERVIEW
    # =====================================================
    if section == "📊 Overview":
        st.header("📊 Overview")

        if "full" in data:
            full_df = data["full"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", f"{len(full_df):,}")
            col2.metric("Fraud Transactions", f"{int(full_df.is_fraud_spike.sum()):,}")
            col3.metric("Test Windows", f"{len(pred_df):,}")
            col4.metric("Detectors Trained", f"{len(det_names)}")

        st.subheader("Results Summary")
        if "metrics" in data:
            metrics_df = data["metrics"]
            st.dataframe(
                metrics_df.style.highlight_max(subset=["f1", "precision", "recall"], color="#2e7d32")
                                .highlight_min(subset=["total_estimated_cost_inr"], color="#2e7d32"),
                use_container_width=True,
            )

            best = metrics_df.loc[metrics_df["f1"].idxmax()]
            st.success(f"🏆 Best detector: **{best['detector']}** — F1={best['f1']:.4f}, "
                       f"Cost=₹{int(best['total_estimated_cost_inr']):,}")

    # =====================================================
    # 2. TRANSACTION TIMELINE
    # =====================================================
    elif section == "📈 Transaction Timeline":
        st.header("📈 Transaction Timeline")
        if "full" not in data:
            st.warning("Full transaction data not found")
            return

        full_df = data["full"].copy()
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
        full_df_ts = full_df.set_index("timestamp").sort_index()

        hourly = full_df_ts.resample("1h").agg(
            total=("amount", "count"),
            fraud=("is_fraud_spike", "sum"),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly["timestamp"], y=hourly["total"],
            mode="lines", fill="tozeroy", name="Total Txns",
            line=dict(color=PALETTE[0], width=1), fillcolor="rgba(79,195,247,0.2)",
        ))
        fig.add_trace(go.Bar(
            x=hourly["timestamp"], y=hourly["fraud"],
            name="Fraud Txns", marker_color="#e57373", opacity=0.8,
        ))
        fig.add_vline(x="2026-07-16", line_dash="dash", line_color=PALETTE[2],
                      annotation_text="Train/Test Split")
        fig.update_layout(
            template="plotly_dark", height=400,
            title="Hourly Transaction Volume (60 Days)",
            xaxis_title="Date", yaxis_title="Transaction Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 3. MODEL COMPARISON
    # =====================================================
    elif section == "🏆 Model Comparison":
        st.header("🏆 Model Comparison")
        if "metrics" not in data:
            st.warning("Metrics not found")
            return

        metrics_df = data["metrics"]

        # Radar chart
        radar_metrics = ["precision", "recall", "f1"]
        if "roc_auc" in metrics_df.columns:
            radar_metrics.append("roc_auc")

        fig = go.Figure()
        for i, (_, row) in enumerate(metrics_df.iterrows()):
            values = [row[m] if pd.notna(row.get(m)) else 0 for m in radar_metrics]
            values.append(values[0])  # close the polygon
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_metrics + [radar_metrics[0]],
                fill="toself",
                name=row["detector"],
                line_color=PALETTE[i % len(PALETTE)],
                opacity=0.7,
            ))
        fig.update_layout(
            template="plotly_dark", height=500,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
            title="Multi-Metric Radar Comparison",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart
        fig2 = px.bar(metrics_df, x="detector", y="f1", color="detector",
                      color_discrete_sequence=PALETTE, template="plotly_dark",
                      title="F1 Score Comparison", text="f1")
        fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    # =====================================================
    # 4. CONFUSION MATRICES
    # =====================================================
    elif section == "🔲 Confusion Matrices":
        st.header("🔲 Confusion Matrices")

        cols = st.columns(min(3, len(det_names)))
        for i, name in enumerate(det_names):
            y_pred = pred_df[f"{name}_pred"].values
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual"),
                            x=["Normal", "Fraud"], y=["Normal", "Fraud"],
                            title=_display_name(name))
            fig.update_layout(template="plotly_dark", height=350, width=350)
            cols[i % len(cols)].plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 5. ROC & PR CURVES
    # =====================================================
    elif section == "📉 ROC & PR Curves":
        st.header("📉 ROC & PR Curves")

        col1, col2 = st.columns(2)

        # ROC
        fig_roc = go.Figure()
        for i, name in enumerate(det_names):
            score_col = f"{name}_score"
            if score_col not in pred_df.columns:
                continue
            fpr, tpr, _ = roc_curve(y_true, pred_df[score_col].values)
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{_display_name(name)} (AUC={roc_auc:.3f})",
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color="gray"), showlegend=False))
        fig_roc.update_layout(template="plotly_dark", title="ROC Curves",
                              xaxis_title="FPR", yaxis_title="TPR", height=450)
        col1.plotly_chart(fig_roc, use_container_width=True)

        # PR
        fig_pr = go.Figure()
        for i, name in enumerate(det_names):
            score_col = f"{name}_score"
            if score_col not in pred_df.columns:
                continue
            prec, rec, _ = precision_recall_curve(y_true, pred_df[score_col].values)
            ap = average_precision_score(y_true, pred_df[score_col].values)
            fig_pr.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines", name=f"{_display_name(name)} (AP={ap:.3f})",
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            ))
        fig_pr.update_layout(template="plotly_dark", title="Precision-Recall Curves",
                             xaxis_title="Recall", yaxis_title="Precision", height=450)
        col2.plotly_chart(fig_pr, use_container_width=True)

    # =====================================================
    # 6. FEATURE IMPORTANCE
    # =====================================================
    elif section == "🎯 Feature Importance":
        st.header("🎯 Feature Importance")
        if "metadata" not in data:
            st.warning("Model metadata not found")
            return

        for det in data["metadata"].get("detectors", []):
            if "top_features" in det:
                st.subheader(det["name"])
                imp_df = pd.DataFrame(
                    list(det["top_features"].items()),
                    columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=True)

                fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Viridis",
                             template="plotly_dark", title=f"{det['name']} — Top Features")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 7. COST TRADEOFF EXPLORER
    # =====================================================
    elif section == "💰 Cost Tradeoff Explorer":
        st.header("💰 Cost Tradeoff Explorer")
        st.markdown("Adjust the cost assumptions to see how the optimal detector changes.")

        col1, col2 = st.columns(2)
        fp_cost = col1.slider("Cost per False Positive (INR)", 10, 5000, 150, step=10)
        fn_cost = col2.slider("Cost per False Negative (INR)", 1000, 100000, 15000, step=1000)

        results = []
        for name in det_names:
            y_pred = pred_df[f"{name}_pred"].values
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            total = fp * fp_cost + fn * fn_cost
            results.append({
                "Detector": _display_name(name),
                "FP": int(fp), "FN": int(fn),
                "FP Cost": int(fp * fp_cost),
                "FN Cost": int(fn * fn_cost),
                "Total Cost (INR)": int(total),
            })
        cost_df = pd.DataFrame(results).sort_values("Total Cost (INR)")
        st.dataframe(cost_df, use_container_width=True)

        fig = px.bar(cost_df, x="Detector", y="Total Cost (INR)", color="Detector",
                     color_discrete_sequence=PALETTE, template="plotly_dark",
                     title=f"Total Estimated Cost (FP=₹{fp_cost}, FN=₹{fn_cost:,})")
        st.plotly_chart(fig, use_container_width=True)

        best = cost_df.iloc[0]
        st.success(f"💰 Lowest cost: **{best['Detector']}** — ₹{int(best['Total Cost (INR)']):,}")

    # =====================================================
    # 8. FAILURE CASE INSPECTOR
    # =====================================================
    elif section == "🔍 Failure Case Inspector":
        st.header("🔍 Failure Case Inspector")

        selected = st.selectbox("Select detector", [_display_name(n) for n in det_names])
        selected_key = det_names[[_display_name(n) for n in det_names].index(selected)]

        pred_col = f"{selected_key}_pred"
        reasons_col = f"{selected_key}_reasons"
        y_pred = pred_df[pred_col].values

        fps = pred_df[(pred_df.window_label == 0) & (pred_df[pred_col] == 1)]
        fns = pred_df[(pred_df.window_label == 1) & (pred_df[pred_col] == 0)]

        col1, col2 = st.columns(2)
        col1.metric("False Positives", len(fps))
        col2.metric("False Negatives", len(fns))

        st.subheader("False Positives (legitimate windows wrongly flagged)")
        if len(fps) > 0:
            display_cols = ["window_start", "txn_count", "unique_devices",
                            "new_device_ratio", "amount_z_max", reasons_col]
            display_cols = [c for c in display_cols if c in fps.columns]
            st.dataframe(fps[display_cols].head(10), use_container_width=True)
        else:
            st.success("No false positives!")

        st.subheader("False Negatives (fraud windows missed)")
        if len(fns) > 0:
            display_cols = ["window_start", "txn_count", "unique_devices",
                            "new_device_ratio", "amount_z_max"]
            display_cols = [c for c in display_cols if c in fns.columns]
            st.dataframe(fns[display_cols].head(10), use_container_width=True)
        else:
            st.success("No false negatives!")


if __name__ == "__main__":
    main()
