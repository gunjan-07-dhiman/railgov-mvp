"""
dashboard/pages/governance_console.py
AI Governance & Operations Console for governed RailGov models.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from app.base import get_service, list_services
from app.config import DATA_DIR
from app.governance.audit import get_events, log_event
from app.governance.drift import detect_drift
from app.governance.registry import get_active, list_versions, load_registry, rollback
from app.supervisor import run_supervisor_cycle
from dashboard.components.charts import create_psi_bar_chart
from dashboard.components.theme import render_header, status_badge


def render_governance_console_page() -> None:
    """Render the engineering and academic AI Governance Console UI."""
    render_header(
        title="Autonomous AI Governance & Operations Console",
        subtitle="Central command center for multi-model drift monitoring, automated supervisor agents, versioning, and immutable audit trails.",
        icon="🛡️",
    )

    # 1. Model Selector
    available_models = list_services()
    if not available_models:
        st.error("No model services are currently registered.")
        return

    selected_model = st.selectbox(
        "Select Governed Model Service",
        available_models,
        index=0,
        help="Choose which AI model to inspect and govern.",
    )
    service = get_service(selected_model)

    active_info = get_active(selected_model)
    active_ver = active_info["version"] if active_info else "Uninitialized"
    model_events = get_events(model_name=selected_model, limit=500)
    versions_list = list_versions(selected_model)

    # Top Metric Cards with real dynamic values
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Active Version", active_ver)
    with m2:
        st.metric("Model Audit Events", len(model_events))
    with m3:
        st.metric("Total Versions", len(versions_list))
    with m4:
        st.metric("Target Variable", service.target_column)

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

    # Tab 1: Model Health
    with tab_health:
        st.subheader(f"Validation & Performance Telemetry: {selected_model}")
        if active_info:
            metrics = active_info.get("metrics", {})
            st.markdown(
                f"**Active Model:** `{active_ver}` | **Training Data:** {active_info.get('training_data_desc', 'N/A')}"
            )

            metric_rows = [
                {"Metric": k.upper(), "Holdout Score": v}
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            ]
            if metric_rows:
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

            perf_alerts = get_events(event_type="PERFORMANCE_ALERT", model_name=selected_model, limit=10)
            if perf_alerts:
                st.markdown("#### ⚠️ Performance Degradation Incident History")
                rows = [
                    {
                        "Timestamp": a["timestamp"],
                        "Baseline": a["details"].get("baseline_auc"),
                        "Current": a["details"].get("current_auc"),
                        "Drop": a["details"].get("auc_drop"),
                    }
                    for a in perf_alerts
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.success("No active performance degradation incidents logged.")
        else:
            st.warning("No active model is registered yet.")

    # Tab 2: Statistical Drift
    with tab_drift:
        st.subheader(f"Statistical Covariate Shift Analysis: {selected_model}")
        st.caption("Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests between baseline and current data batches.")

        drifted_file = DATA_DIR / "drifted.csv"
        try:
            base_df = service.load_baseline_df()
            if drifted_file.exists():
                batch_df = pd.read_csv(drifted_file)
                drift_report = detect_drift(
                    base_df,
                    batch_df,
                    feature_columns=service.feature_columns,
                    categorical_features=service.categorical_features,
                    numeric_features=service.numeric_features,
                )

                severity = drift_report["overall_severity"]
                st.markdown(f"### Overall Drift: {status_badge(severity)} ({drift_report['critical_count']} Critical Features)")

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
            else:
                st.info("Production batch data (data/drifted.csv) not found for drift comparison.")
        except Exception as exc:
            st.error(f"Drift assessment error: {exc}")

    # Tab 3: Model Registry & Rollback
    with tab_reg:
        st.subheader(f"Model Version Catalog: {selected_model}")
        reg_data = load_registry()
        model_cat = reg_data.get("models", {}).get(selected_model, {})
        curr_ver = model_cat.get("active_version")
        prev_ver = model_cat.get("previous_active_version")

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**Current Active Model:** `{curr_ver}` | **Previous Version:** `{prev_ver or 'None'}`")
        with c2:
            if st.button("⏪ Rollback Model", disabled=not prev_ver, type="secondary"):
                try:
                    updated = rollback(model_name=selected_model)
                    log_event(
                        event_type="ROLLBACK",
                        actor="governance_ui",
                        details={"reverted_from": curr_ver, "reverted_to": updated["active_version"]},
                        model_version=updated["active_version"],
                        model_name=selected_model,
                    )
                    st.success(f"Rolled back {selected_model} to {updated['active_version']}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Rollback failed: {e}")

        v_rows = [
            {
                "Version": v.get("version"),
                "Status": v.get("status", "").upper(),
                "Metrics": json.dumps(v.get("metrics", {})),
                "Created At": v.get("created_at"),
                "Training Description": v.get("training_data_desc"),
            }
            for v in versions_list
        ]
        st.dataframe(pd.DataFrame(v_rows), use_container_width=True)

    # Tab 4: Immutable Audit Trail
    with tab_audit:
        st.subheader(f"Immutable Audit Trail (models/audit.db)")
        st.caption("Backed by SQLite triggers aborting UPDATE or DELETE operations to guarantee non-repudiation.")

        filter_type = st.selectbox(
            "Filter Event Type",
            ["ALL", "PREDICTION", "DATA_QUALITY_BLOCK", "DRIFT_ALERT", "PERFORMANCE_ALERT", "RETRAIN_STARTED", "MODEL_PROMOTED", "MODEL_REJECTED", "ROLLBACK", "SUPERVISOR_DECISION"],
        )
        selected_type = None if filter_type == "ALL" else filter_type
        events = get_events(event_type=selected_type, model_name=selected_model, limit=100)

        ev_rows = [
            {
                "ID": e["id"],
                "Timestamp": e["timestamp"],
                "Event Type": e["event_type"],
                "Actor": e["actor"],
                "Model": e.get("model_name", selected_model),
                "Version": e["model_version"],
                "Details": json.dumps(e["details"]),
            }
            for e in events
        ]
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True)

    # Tab 5: Autonomous Supervisor
    with tab_sup:
        st.subheader(f"Autonomous Rule-Based Supervisor: {selected_model}")
        st.caption("Monitors telemetry, performs root cause analysis, executes retrain cycles, and escalates failures.")

        if st.button("⚡ Run Autonomous Incident Resolution Cycle", type="primary"):
            drifted_file = DATA_DIR / "drifted.csv"
            if drifted_file.exists():
                batch_df = pd.read_csv(drifted_file)
                with st.spinner(f"Supervisor executing governance cycle for {selected_model}..."):
                    report = run_supervisor_cycle(batch_df, service=service)
                st.success("Governance Incident Cycle Completed!")
                st.json(report)
                st.rerun()
            else:
                st.error("data/drifted.csv not found.")

        sup_events = get_events(event_type="SUPERVISOR_DECISION", model_name=selected_model, limit=10)
        if sup_events:
            st.markdown("#### Recent Autonomous Decisions")
            for se in sup_events:
                det = se.get("details", {})
                priority = det.get("priority", "N/A")
                action = det.get("action", "N/A")
                with st.expander(f"Incident {se['id']} - [{priority}] Action: {action} ({se['timestamp']})"):
                    st.write(f"**Root Cause Diagnosis:** {det.get('root_cause')}")
                    st.write(f"**Remediation Action:** {det.get('resolution')}")
                    if det.get("drift_summary"):
                        st.json(det["drift_summary"])
