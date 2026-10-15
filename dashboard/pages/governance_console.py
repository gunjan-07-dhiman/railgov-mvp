"""
dashboard/pages/governance_console.py
AI Governance & Operations Console for governed YatraGov AI models.
Theme: "Departure Board & Ticket" (Control-Room Identity).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from app.base import get_service, list_services
import app.modules.ticket_predictor  # Auto-registers ticket predictor service
import app.modules.delay             # Auto-registers delay predictor service
from app.config import APP_NAME
from app.governance.audit import get_events, log_event
from app.governance.drift import detect_drift
from app.governance.registry import get_active, list_versions, load_registry, rollback
from app.supervisor import run_supervisor_cycle
from dashboard.components.charts import create_psi_bar_chart
from dashboard.components.theme import apply_theme
from dashboard.components.ui import (
    render_footer,
    render_page_header,
    render_sidebar_brand,
    render_sidebar_route_navigation,
    render_sidebar_status_strip,
    signal_light,
)

st.set_page_config(
    page_title=f"{APP_NAME} - AI Governance Console",
    page_icon="🚆",
    layout="wide",
)


def main():
    """Render the engineering and academic AI Governance Console UI."""
    apply_theme()

    available_models = list_services()
    if not available_models:
        st.error("No model services are currently registered.")
        return

    # Dark navy sidebar
    render_sidebar_brand()
    render_sidebar_route_navigation(active_page_id="governance")

    # Page Header in Barlow Condensed with saffron underline
    render_page_header(
        title="AI GOVERNANCE & OPERATIONS CONSOLE",
        subtitle="Real-time covariate drift monitoring (PSI/KS), model catalog, rollback controls, and autonomous supervisor logs.",
    )

    col_sel, col_stat = st.columns([2, 3])
    with col_sel:
        selected_model = st.selectbox(
            "Select Governed Model Service",
            available_models,
            index=0,
            help="Choose which AI model to inspect and govern.",
        )
    service = get_service(selected_model)

    active_info = get_active(selected_model)
    active_ver = active_info["version"] if active_info else "Uninitialized"
    render_sidebar_status_strip(active_ver)

    model_events = get_events(model_name=selected_model, limit=500)
    versions_list = list_versions(selected_model)

    # Control-Room Top KPI Panels (Dark Navy)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div style="background-color: #0B1F3A; border: 1px solid #1E3A63; border-radius: 8px; padding: 0.85rem 1rem; color: #FFFFFF;">
                <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.72rem; color: #8FA3BF; text-transform: uppercase; letter-spacing: 0.1em;">ACTIVE VERSION</div>
                <div style="font-family: 'Share Tech Mono', monospace; font-size: 1.55rem; color: #FF9F1C; font-weight: 700;">{active_ver}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div style="background-color: #0B1F3A; border: 1px solid #1E3A63; border-radius: 8px; padding: 0.85rem 1rem; color: #FFFFFF;">
                <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.72rem; color: #8FA3BF; text-transform: uppercase; letter-spacing: 0.1em;">AUDIT EVENTS</div>
                <div style="font-family: 'Share Tech Mono', monospace; font-size: 1.55rem; color: #FFFFFF; font-weight: 700;">{len(model_events)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div style="background-color: #0B1F3A; border: 1px solid #1E3A63; border-radius: 8px; padding: 0.85rem 1rem; color: #FFFFFF;">
                <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.72rem; color: #8FA3BF; text-transform: uppercase; letter-spacing: 0.1em;">TOTAL VERSIONS</div>
                <div style="font-family: 'Share Tech Mono', monospace; font-size: 1.55rem; color: #FFFFFF; font-weight: 700;">{len(versions_list)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div style="background-color: #0B1F3A; border: 1px solid #1E3A63; border-radius: 8px; padding: 0.85rem 1rem; color: #FFFFFF;">
                <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.72rem; color: #8FA3BF; text-transform: uppercase; letter-spacing: 0.1em;">TARGET VARIABLE</div>
                <div style="font-family: 'Share Tech Mono', monospace; font-size: 1.3rem; color: #FF9F1C; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{service.target_column}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-top: 1.25rem;"></div>', unsafe_allow_html=True)

    tab_health, tab_drift, tab_reg, tab_audit, tab_sup = st.tabs(
        [
            "Model Health",
            "Statistical Drift (PSI/KS)",
            "Model Registry & Rollback",
            "Immutable Audit Trail",
            "Autonomous Supervisor",
        ]
    )

    # Tab 1: Model Health
    with tab_health:
        st.markdown(f'<div class="rail-section-header">TELEMETRY & VALIDATION BENCHMARKS: {selected_model.upper()}</div>', unsafe_allow_html=True)
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
                st.markdown("#### Performance Degradation Incident History")
                rows = [
                    {
                        "Timestamp": a["timestamp"],
                        "Baseline": a["details"].get("baseline_auc") or a["details"].get("baseline_mae"),
                        "Current": a["details"].get("current_auc") or a["details"].get("current_mae"),
                        "Drop": a["details"].get("auc_drop") or a["details"].get("mae_increase"),
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
        st.markdown(f'<div class="rail-section-header">COVARIATE DISTRIBUTION SHIFT: {selected_model.upper()}</div>', unsafe_allow_html=True)
        st.caption("Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests between baseline and current data batches.")

        try:
            base_df = service.load_baseline_df()
            batch_df = service.load_drifted_df()
            drift_report = detect_drift(
                base_df,
                batch_df,
                feature_columns=service.feature_columns,
                categorical_features=service.categorical_features,
                numeric_features=service.numeric_features,
            )

            severity = drift_report["overall_severity"]
            st.markdown(
                f"**Overall Track Drift:** {signal_light(severity, severity.upper())} ({drift_report['critical_count']} Critical Features)",
                unsafe_allow_html=True,
            )

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
        except Exception as exc:
            st.error(f"Drift assessment error: {exc}")

    # Tab 3: Model Registry & Rollback
    with tab_reg:
        st.markdown(f'<div class="rail-section-header">VERSION CATALOG & DEPLOYMENT: {selected_model.upper()}</div>', unsafe_allow_html=True)
        reg_data = load_registry()
        model_cat = reg_data.get("models", {}).get(selected_model, {})
        curr_ver = model_cat.get("active_version")
        prev_ver = model_cat.get("previous_active_version")

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**Current Active Model:** `{curr_ver}` | **Previous Version:** `{prev_ver or 'None'}`")
        with c2:
            if st.button("ROLLBACK MODEL", disabled=not prev_ver, type="secondary"):
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
        st.markdown('<div class="rail-section-header">IMMUTABLE AUDIT LOG (MODELS/AUDIT.DB)</div>', unsafe_allow_html=True)
        st.caption("Backed by SQLite abort triggers preventing UPDATE or DELETE operations to guarantee non-repudiation.")

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
        st.markdown(f'<div class="rail-section-header">AUTONOMOUS SUPERVISOR CONTROLLER: {selected_model.upper()}</div>', unsafe_allow_html=True)
        st.caption("Monitors telemetry, performs root cause analysis, executes retrain cycles, and escalates failures.")

        if st.button("RUN INCIDENT RESOLUTION CYCLE", type="primary"):
            try:
                batch_df = service.load_drifted_df()
                with st.spinner(f"Supervisor executing governance cycle for {selected_model}..."):
                    report = run_supervisor_cycle(batch_df, service=service)
                st.success("Governance Incident Cycle Completed!")
                st.json(report)
                st.rerun()
            except Exception as e:
                st.error(f"Supervisor execution failed: {e}")

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

    render_footer()


if __name__ == "__main__":
    main()
