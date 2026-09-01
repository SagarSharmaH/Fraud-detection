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
  8. Spike Type Analysis          — Detection rates per attack type
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Any

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

log = logging.getLogger(__name__)

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


def _get_detector_columns(df: pd.DataFrame) -> list[str]:
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    return [c.replace("_pred", "") for c in pred_cols]


def _display_name(name: str) -> str:
    return name.replace("_", " ").title()


@st.cache_data
def load_data() -> dict[str, Any]:
    data: dict[str, Any] = {}
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


def main() -> None:
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
        "🧬 Spike Type Analysis",
        "📤 Upload & Score CSV",
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

            # Highlight columns that exist
            highlight_max_cols = [c for c in ["f1", "precision", "recall", "mcc"] if c in metrics_df.columns]
            highlight_min_cols = [c for c in ["total_estimated_cost_inr"] if c in metrics_df.columns]

            styled = metrics_df.style
            if highlight_max_cols:
                styled = styled.highlight_max(subset=highlight_max_cols, color="#2e7d32")
            if highlight_min_cols:
                styled = styled.highlight_min(subset=highlight_min_cols, color="#2e7d32")

            st.dataframe(styled, use_container_width=True)

            best = metrics_df.loc[metrics_df["f1"].idxmax()]
            mcc_str = f", MCC={best['mcc']:.4f}" if "mcc" in metrics_df.columns else ""
            st.success(f"🏆 Best detector: **{best['detector']}** — F1={best['f1']:.4f}{mcc_str}, "
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
        if "mcc" in metrics_df.columns:
            radar_metrics.append("mcc")

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

    # =====================================================
    # 9. SPIKE TYPE ANALYSIS
    # =====================================================
    elif section == "🧬 Spike Type Analysis":
        st.header("🧬 Spike Type Analysis")

        if "spike_type" not in pred_df.columns:
            st.warning("Spike type data not available. Re-run the pipeline to generate it.")
            return

        fraud_df = pred_df[pred_df.window_label == 1].copy()
        if len(fraud_df) == 0:
            st.warning("No fraud windows in the test set.")
            return

        spike_types = [st_val for st_val in fraud_df["spike_type"].unique() if st_val != "none"]
        if not spike_types:
            st.info("No labeled spike types found.")
            return

        # Build detection rate matrix
        rates_data: list[dict[str, Any]] = []
        for name in det_names:
            pred_col = f"{name}_pred"
            for st_val in spike_types:
                mask = fraud_df["spike_type"] == st_val
                rate = float(fraud_df.loc[mask, pred_col].mean()) if mask.sum() > 0 else 0.0
                rates_data.append({
                    "Detector": _display_name(name),
                    "Spike Type": st_val,
                    "Detection Rate": rate,
                    "Count": int(mask.sum()),
                })

        rates_df = pd.DataFrame(rates_data)

        # Heatmap
        pivot = rates_df.pivot(index="Detector", columns="Spike Type", values="Detection Rate")
        fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="RdYlGn",
                        labels=dict(color="Detection Rate"), template="plotly_dark",
                        title="Detection Rate by Spike Type & Detector")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart
        fig2 = px.bar(rates_df, x="Spike Type", y="Detection Rate", color="Detector",
                      barmode="group", color_discrete_sequence=PALETTE,
                      template="plotly_dark",
                      title="Detection Rate Breakdown by Attack Type")
        st.plotly_chart(fig2, use_container_width=True)

        # Summary stats
        st.subheader("Spike Type Distribution")
        type_counts = fraud_df["spike_type"].value_counts()
        st.dataframe(type_counts.reset_index().rename(
            columns={"index": "Spike Type", "spike_type": "Spike Type", "count": "Windows"}
        ), use_container_width=True)

    # =====================================================
    # 10. UPLOAD & SCORE CSV
    # =====================================================
    elif section == "📤 Upload & Score CSV":
        st.header("📤 Upload & Score New Payment Transactions")
        st.markdown("""
        Upload a CSV file of new merchant payment transactions to run real-time feature engineering,
        multi-model fraud detection, and explainable AI reason code generation.
        """)

        # Sample format info
        with st.expander("ℹ️ Required CSV Format & Example"):
            st.markdown("""
            Your uploaded CSV must contain the following 5 columns:
            - `txn_id`: Unique transaction string (e.g. `tx_99901`)
            - `timestamp`: Date and time (e.g. `2026-08-01 14:30:00`)
            - `amount`: Numeric transaction value (e.g. `250.00`)
            - `device_id`: Device or IP identifier (e.g. `dev_0123`)
            - `geo`: Location string (e.g. `Mumbai`)
            """)
            sample_df = pd.DataFrame([
                {"txn_id": "tx_001", "timestamp": "2026-08-01 10:00:00", "amount": 150.00, "device_id": "dev_0001", "geo": "Mumbai"},
                {"txn_id": "tx_002", "timestamp": "2026-08-01 10:00:15", "amount": 95000.00, "device_id": "dev_0001", "geo": "Mumbai"},
            ])
            st.dataframe(sample_df, use_container_width=True)

        uploaded_file = st.file_uploader("Choose a transaction CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Loaded {len(raw_df):,} transaction records.")
                st.dataframe(raw_df.head(5), use_container_width=True)

                req_cols = ["txn_id", "timestamp", "amount", "device_id", "geo"]
                missing = [c for c in req_cols if c not in raw_df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                    return

                if "is_fraud_spike" not in raw_df.columns:
                    raw_df["is_fraud_spike"] = 0

                selected_det_name = st.selectbox(
                    "Select Risk Model for Scoring",
                    [_display_name(n) for n in det_names]
                )
                selected_key = det_names[[_display_name(n) for n in det_names].index(selected_det_name)]

                if st.button("🚀 Run Risk Engine & Score Transactions"):
                    with st.spinner("Engineering 16 rolling features & running risk model..."):
                        from features import build_window_features, _historical_stats
                        from model_store import load_all_models

                        models_list = load_all_models("models")
                        target_model = None
                        for m in models_list:
                            safe = m.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                            if safe == selected_key or m.name == selected_det_name:
                                target_model = m
                                break

                        if target_model is None and models_list:
                            target_model = models_list[0]

                        # Build baseline stats from train data if available
                        if "train" in data:
                            hist_stats = _historical_stats(data["train"])
                        else:
                            hist_stats = _historical_stats(raw_df)

                        window_feats = build_window_features(raw_df, hist_stats)

                        if len(window_feats) == 0:
                            st.warning("No valid time windows generated from the uploaded data.")
                            return

                        preds, scores, reasons = target_model.predict(window_feats)

                        window_feats["predicted_label"] = preds
                        window_feats["risk_score"] = scores
                        window_feats["xai_reasons"] = [", ".join(r) if r else "Normal" for r in reasons]

                        n_flagged = int(preds.sum())
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Scored Time Windows", len(window_feats))
                        col2.metric("Flagged Fraud Windows", n_flagged, delta_color="inverse")
                        col3.metric("Anomaly Rate", f"{(n_flagged / len(window_feats) * 100):.1f}%")

                        st.subheader("Scored Risk Results")
                        st.dataframe(
                            window_feats[["window_start", "txn_count", "amount_max", "predicted_label", "risk_score", "xai_reasons"]],
                            use_container_width=True
                        )

                        # CSV Download
                        csv_output = window_feats.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Scored Risk Results CSV",
                            data=csv_output,
                            file_name="sentinelrisk_scored_predictions.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    main()

