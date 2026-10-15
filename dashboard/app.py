"""
dashboard/app.py
Streamlit Application for RailGov MVP.
Features two distinct operational modes:
1. 🎫 Passenger Portal: Clean, intuitive, consumer-facing ticket confirmation checker.
2. 🛡️ AI Governance Console: Academic defense & ML engineering console (Drift, Registry, Audit, Supervisor).
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path when executed via streamlit run
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import (
    DATA_DIR,
    PSI_CRITICAL_THRESHOLD,
    PSI_WARNING_THRESHOLD,
    VALID_CLASSES,
    VALID_STATUSES,
    VALID_TRAIN_TYPES,
)
from app.governance.audit import get_events, init_audit_db, log_event
from app.governance.data_quality import validate_record
from app.governance.drift import detect_drift
from app.governance.registry import get_active, list_versions, load_registry, rollback
from app.predictor import load_model, predict
from app.supervisor import run_supervisor_cycle

# Page configuration
st.set_page_config(
    page_title="RailGov - Indian Railways AI Predictor & Governance",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_active_bundle():
    """Load active model bundle and metadata. Bootstraps v1 if registry is empty."""
    active = get_active()
    if not active:
        baseline_path = DATA_DIR / "baseline.csv"
        if baseline_path.exists():
            from app.governance.registry import register
            from app.predictor import train
            base_df = pd.read_csv(baseline_path)
            res = train(base_df, version="v1")
            register(
                version="v1",
                metrics=res["metrics"],
                training_data_desc="Bootstrap baseline model v1",
                status="active",
            )
            active = get_active()
        else:
            return None, None
    try:
        bundle, meta = load_model(active["version"])
        return bundle, meta
    except Exception as e:
        st.error(f"Failed to load active model '{active.get('version')}': {e}")
        return None, None


def create_psi_bar_chart(drift_report: Dict[str, Any]) -> go.Figure:
    """Generate a Plotly bar chart of feature PSI values with severity color coding."""
    feature_metrics = drift_report.get("feature_metrics", {})
    if not feature_metrics:
        fig = go.Figure()
        fig.update_layout(title="No drift metrics available.")
        return fig

    features = list(feature_metrics.keys())
    psi_values = [feature_metrics[f]["psi"] for f in features]

    colors = []
    for psi in psi_values:
        if psi >= PSI_CRITICAL_THRESHOLD:
            colors.append("#ef4444")  # Red: Critical
        elif psi >= PSI_WARNING_THRESHOLD:
            colors.append("#f97316")  # Orange: Warning
        else:
            colors.append("#10b981")  # Green: Normal

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=features,
            y=psi_values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in psi_values],
            textposition="auto",
        )
    )

    # Reference lines for Warning and Critical PSI thresholds
    fig.add_hline(
        y=PSI_WARNING_THRESHOLD,
        line_dash="dash",
        line_color="#f97316",
        annotation_text="Warning (0.10)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=PSI_CRITICAL_THRESHOLD,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Critical (0.20)",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Population Stability Index (PSI) per Feature",
        xaxis_title="Feature",
        yaxis_title="PSI Value",
        yaxis=dict(range=[0, max(max(psi_values) * 1.25, 0.35)]),
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def render_passenger_portal(bundle, active_ver: str):
    """Clean, intuitive passenger-facing ticket prediction interface."""
    st.markdown("## 🎫 Indian Railways Ticket Confirmation Predictor")
    st.markdown("Check whether your Waitlisted (WL) or RAC ticket will get confirmed before chart preparation.")

    st.markdown("---")

    col_form, col_space, col_result = st.columns([1.2, 0.1, 1.2])

    with col_form:
        st.markdown("### 📝 Enter Ticket Details")

        class_labels = {
            "3A": "3rd AC (3A) - Three Tier",
            "SL": "Sleeper Class (SL)",
            "2A": "2nd AC (2A) - Two Tier",
            "1A": "1st AC (1A) - Executive",
            "CC": "AC Chair Car (CC)",
        }
        travel_class = st.selectbox(
            "Travel Class",
            VALID_CLASSES,
            format_func=lambda c: class_labels.get(c, c),
            index=0,
        )

        status_labels = {
            "WL": "Waitlist (WL)",
            "RAC": "Reservation Against Cancellation (RAC)",
        }
        booking_status = st.radio(
            "Current Booking Status",
            VALID_STATUSES,
            format_func=lambda s: status_labels.get(s, s),
            horizontal=True,
        )

        max_pos = 60 if booking_status == "RAC" else 200
        wl_pos = st.number_input(
            f"Current Queue Position ({booking_status} Number)",
            min_value=1,
            max_value=max_pos,
            value=min(18, max_pos),
            help=f"Enter your current ticket status number (1 to {max_pos}).",
        )

        days_before = st.slider(
            "Days Remaining Before Journey",
            min_value=0,
            max_value=120,
            value=25,
            help="Number of days remaining until train departure date.",
        )

        with st.expander("Additional Journey Details (Optional)", expanded=False):
            train_type = st.selectbox("Train Category", VALID_TRAIN_TYPES, index=1)
            month = st.selectbox(
                "Journey Month",
                list(range(1, 13)),
                format_func=lambda m: [
                    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
                ][m - 1],
                index=9,
            )
            dow = st.selectbox(
                "Day of Week",
                list(range(7)),
                format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d],
                index=4,
            )
            festival_rush = st.checkbox("Diwali / Chhath / Holiday Festival Rush", value=False)
            route_pop = st.slider("Route Demand / Popularity", 0.0, 1.0, 0.60, 0.05)

        predict_button = st.button("🔮 Check Confirmation Chance", use_container_width=True, type="primary")

    with col_result:
        st.markdown("### 📊 Prediction Result")

        if predict_button:
            # Build record
            record = {
                "travel_class": travel_class,
                "booking_status": booking_status,
                "wl_position": int(wl_pos),
                "days_before_journey": int(days_before),
                "day_of_week": int(dow),
                "month": int(month),
                "is_festival_season": 1 if festival_rush else 0,
                "route_popularity": float(route_pop),
                "train_type": train_type,
            }

            # Pre-inference validation
            is_valid, issues = validate_record(record)
            if not is_valid:
                st.error(f"Invalid Ticket Details: {', '.join(issues)}")
                log_event("DATA_QUALITY_BLOCK", actor="passenger_portal", details={"record": record, "issues": issues})
            elif not bundle:
                st.error("Ticket prediction engine is initializing. Please retry in a few seconds.")
            else:
                res = predict(record, bundle)
                prob = res["probability"]
                prob_pct = prob * 100.0
                status_str = res["predicted_status"]

                # Log prediction to immutable audit trail
                log_event("PREDICTION", actor="passenger_portal", details={"ticket": record, "prediction": res}, model_version=active_ver)

                # Visual result card
                if status_str == "Likely Confirmed":
                    st.success("### 🟢 HIGH CHANCE OF CONFIRMATION")
                    status_desc = "Your ticket has a high likelihood of getting confirmed before chart preparation."
                elif status_str == "Uncertain":
                    st.warning("### 🟡 MODERATE / UNCERTAIN CHANCE")
                    status_desc = "Confirmation depends on cancellations in the final 24-48 hours before departure."
                else:
                    st.error("### 🔴 LOW CHANCE OF CONFIRMATION")
                    status_desc = "Unlikely to confirm. We strongly recommend booking Tatkal or an alternate train."

                st.markdown(f"**Confirmation Probability:** `{prob_pct:.1f}%` ({res['confidence']} Confidence)")
                st.info(status_desc)

                st.markdown("#### 💡 Why this prediction?")
                for factor in res["top_factors"]:
                    st.markdown(f"- **{factor['explanation']}**")
        else:
            st.info("👈 Fill your ticket details on the left and click **'Check Confirmation Chance'** to see AI predictions.")


def render_governance_console(bundle, active_info):
    """Engineering and academic defense console for AI governance."""
    st.markdown("## 🛡️ Autonomous AI Governance & Operations Console")
    st.caption("Central command center governing ticket prediction models with automated drift detection, model versioning, self-healing retrain cycles, and immutable audit logs.")

    # High-level telemetry
    active_ver = active_info["version"] if active_info else "Uninitialized"
    recent_events = get_events(limit=500)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Active Model Version", active_ver)
    with m2:
        st.metric("Total Audit Events", len(recent_events))
    with m3:
        st.metric("Data Quality Guardrail", "ACTIVE (Enforced)")
    with m4:
        st.metric("Supervisor Agent", "STANDBY / ACTIVE")

    st.markdown("---")

    tab_health, tab_drift, tab_reg, tab_audit, tab_sup = st.tabs(
        [
            "📊 Model Health",
            "🌊 Statistical Drift (PSI/KS)",
            "📦 Model Registry & Rollback",
            "📜 Immutable Audit Trail",
            "🤖 Autonomous Supervisor",
        ]
    )

    # 1. Model Health
    with tab_health:
        st.subheader("Model Validation & Accuracy Telemetry")
        if active_info:
            metrics = active_info.get("metrics", {})
            st.markdown(f"**Active Model:** `{active_info.get('version')}` | **Training Data:** {active_info.get('training_data_desc')}")
            
            m_df = pd.DataFrame([
                {"Metric": "ROC-AUC", "Holdout Value": metrics.get("auc"), "Threshold Target": ">= 0.80"},
                {"Metric": "Accuracy", "Holdout Value": metrics.get("accuracy"), "Threshold Target": ">= 0.70"},
                {"Metric": "F1-Score", "Holdout Value": metrics.get("f1"), "Threshold Target": ">= 0.70"},
            ])
            st.dataframe(m_df, use_container_width=True, hide_index=True)

            perf_alerts = get_events(event_type="PERFORMANCE_ALERT", limit=10)
            if perf_alerts:
                st.markdown("#### ⚠️ Performance Degradation Incident History")
                rows = [
                    {
                        "Timestamp": a["timestamp"],
                        "Baseline AUC": a["details"].get("baseline_auc"),
                        "Degraded AUC": a["details"].get("current_auc"),
                        "Drop": a["details"].get("auc_drop"),
                    }
                    for a in perf_alerts
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.success("No performance degradation alerts currently active.")

    # 2. Statistical Drift
    with tab_drift:
        st.subheader("Statistical Covariate Shift Analysis")
        st.write("Measures Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests between baseline and production batches.")

        baseline_file = DATA_DIR / "baseline.csv"
        drifted_file = DATA_DIR / "drifted.csv"

        if baseline_file.exists() and drifted_file.exists():
            base_df = pd.read_csv(baseline_file)
            drift_df = pd.read_csv(drifted_file)
            drift_report = detect_drift(base_df, drift_df)

            severity = drift_report["overall_severity"].upper()
            color = "🔴" if severity == "CRITICAL" else ("🟠" if severity == "WARNING" else "🟢")
            st.markdown(f"### Overall Drift: {color} **{severity}** ({drift_report['critical_count']} Critical Features)")

            fig = create_psi_bar_chart(drift_report)
            st.plotly_chart(fig, use_container_width=True)

            breakdown = [
                {
                    "Feature": k,
                    "Type": v["feature_type"],
                    "PSI": v["psi"],
                    "KS Statistic": v.get("ks_statistic"),
                    "KS p-value": v.get("ks_pvalue"),
                    "Severity": v["severity"].upper(),
                }
                for k, v in drift_report["feature_metrics"].items()
            ]
            st.dataframe(pd.DataFrame(breakdown), use_container_width=True)

    # 3. Model Registry & Rollback
    with tab_reg:
        st.subheader("Model Lifecycle Registry (models/registry.json)")
        reg_data = load_registry()
        versions = reg_data.get("versions", [])
        curr_ver = reg_data.get("active_version")
        prev_ver = reg_data.get("previous_active_version")

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**Current Active Model:** `{curr_ver}` | **Previous Active Model:** `{prev_ver}`")
        with c2:
            if st.button("⏪ Rollback to Previous Version", disabled=not prev_ver, type="secondary"):
                try:
                    updated = rollback()
                    log_event(
                        event_type="ROLLBACK",
                        actor="governance_ui",
                        details={"reverted_from": curr_ver, "reverted_to": updated["active_version"]},
                        model_version=updated["active_version"],
                    )
                    st.success(f"Successfully rolled back to {updated['active_version']}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Rollback failed: {e}")

        v_rows = [
            {
                "Version": v.get("version"),
                "Status": v.get("status").upper(),
                "AUC": v.get("metrics", {}).get("auc"),
                "Created At": v.get("created_at"),
                "Dataset Description": v.get("training_data_desc"),
            }
            for v in versions
        ]
        st.dataframe(pd.DataFrame(v_rows), use_container_width=True)

    # 4. Audit Trail
    with tab_audit:
        st.subheader("Append-Only Audit Log (models/audit.db)")
        st.caption("Guaranteed non-repudiation: database triggers automatically abort UPDATE or DELETE operations.")

        filter_type = st.selectbox(
            "Filter Audit Records",
            ["ALL", "PREDICTION", "DATA_QUALITY_BLOCK", "DRIFT_ALERT", "PERFORMANCE_ALERT", "RETRAIN_STARTED", "MODEL_PROMOTED", "MODEL_REJECTED", "ROLLBACK", "SUPERVISOR_DECISION"],
        )
        selected = None if filter_type == "ALL" else filter_type
        events = get_events(event_type=selected, limit=100)

        ev_rows = [
            {
                "ID": e["id"],
                "Timestamp": e["timestamp"],
                "Event Type": e["event_type"],
                "Actor": e["actor"],
                "Model Version": e["model_version"],
                "Details JSON": json.dumps(e["details"]),
            }
            for e in events
        ]
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True)

    # 5. Autonomous Supervisor
    with tab_sup:
        st.subheader("Autonomous Rule-Based Supervisor Agent")
        st.write("Orchestrates self-healing pipelines: detects drift, diagnoses root causes, executes retrain cycles, and escalates if new models fail validation.")

        if st.button("⚡ Run Autonomous Incident Resolution Cycle", type="primary"):
            drifted_file = DATA_DIR / "drifted.csv"
            if drifted_file.exists():
                batch_df = pd.read_csv(drifted_file)
                with st.spinner("Supervisor analyzing drift, evaluating degradation, and retraining model..."):
                    report = run_supervisor_cycle(batch_df)
                st.success("Incident Resolution Cycle Completed!")
                st.json(report)
                st.rerun()
            else:
                st.error("data/drifted.csv not found.")

        sup_events = get_events(event_type="SUPERVISOR_DECISION", limit=10)
        if sup_events:
            st.markdown("#### Recent Supervisor Decisions")
            for se in sup_events:
                det = se.get("details", {})
                priority = det.get("priority", "N/A")
                action = det.get("action", "N/A")
                with st.expander(f"Incident {se['id']} - [{priority}] Action: {action} ({se['timestamp']})"):
                    st.write(f"**Root Cause Diagnosis:** {det.get('root_cause')}")
                    st.write(f"**Remediation Action:** {det.get('resolution')}")
                    if det.get("drift_summary"):
                        st.json(det["drift_summary"])


def main():
    init_audit_db()
    bundle, meta = load_active_bundle()
    active_info = get_active()
    active_ver = active_info["version"] if active_info else "v1"

    # Sidebar: Mode Selector
    st.sidebar.title("🚆 RailGov System")
    portal_mode = st.sidebar.radio(
        "Select Portal Mode",
        [
            "🎫 Passenger Portal (Simple)",
            "🛡️ AI Governance Console (Defense / Admin)",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 System Info")
    st.sidebar.markdown(f"- **Active Model**: `{active_ver}`")
    st.sidebar.markdown("- **Engine**: GradientBoosting")
    st.sidebar.markdown("- **Explainability**: TreeSHAP (Log-odds)")
    st.sidebar.markdown("- **Audit Engine**: SQLite (Append-Only)")

    if "Passenger Portal" in portal_mode:
        render_passenger_portal(bundle, active_ver)
    else:
        render_governance_console(bundle, active_info)


if __name__ == "__main__":
    main()
