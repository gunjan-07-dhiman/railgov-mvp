"""
dashboard/pages/fraud.py
Booking Fraud Shield & Bot Detection Console for YatraGov AI.
Theme: "Departure Board & Ticket" (Indian Railway Signage & Warm Paper).
"""

from typing import Any, Dict
import pandas as pd
import streamlit as st

from app.base import get_service
from app.config import APP_NAME
from app.governance.audit import get_events
from dashboard.components.charts import create_fraud_risk_gauge
from dashboard.components.theme import apply_theme
from dashboard.components.ui import (
    departure_board,
    load_active_bundle,
    render_footer,
    render_page_header,
    render_sidebar_brand,
    render_sidebar_route_navigation,
    render_sidebar_status_strip,
    signal_light,
)

st.set_page_config(
    page_title=f"{APP_NAME} - Booking Fraud Shield",
    page_icon="🚆",
    layout="wide",
)


def main():
    """Render Booking Fraud Shield and Session Assessment UI."""
    apply_theme()

    # Dark navy sidebar
    render_sidebar_brand()
    render_sidebar_route_navigation(active_page_id="fraud")

    service = get_service("fraud_detector")
    bundle, meta = load_active_bundle("fraud_detector")
    active_ver = meta.get("version", "v1") if isinstance(meta, dict) else "v1"
    render_sidebar_status_strip(active_ver=active_ver)

    # Page Header in Barlow Condensed with saffron underline
    render_page_header(
        title="ANTI-CHEAT / ANTI-BOT SYSTEM",
        subtitle="Catches scalpers and automated bots booking all the Tatkal tickets in 5 seconds.",
    )

    st.markdown(
        '<div style="font-family: \'Barlow Condensed\', sans-serif; font-size: 0.78rem; color: #8FA3BF; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.75rem;">'
        'Demo data - Academic simulation of high-velocity passenger booking sessions.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_form, col_space, col_result = st.columns([1.15, 0.05, 1.15])

    with col_form:
        st.markdown('<div class="rail-section-header">CHECK A BOOKING SESSION</div>', unsafe_allow_html=True)

        with st.form("fraud_check_form"):
            col1, col2 = st.columns(2)
            with col1:
                form_fill_sec = st.number_input(
                    "Form Fill Duration (seconds)",
                    min_value=0.5,
                    max_value=600.0,
                    value=24.5,
                    step=0.5,
                    help="Elapsed time taken to populate passenger credentials and route choices.",
                )
                bookings_last_hr = st.number_input(
                    "Bookings in Past Hour",
                    min_value=0,
                    max_value=100,
                    value=1,
                    step=1,
                    help="Aggregate reservation requests submitted by this session identity in the last 60 minutes.",
                )

            with col2:
                accounts_per_dev = st.number_input(
                    "Accounts per Device Profile",
                    min_value=1,
                    max_value=30,
                    value=1,
                    step=1,
                    help="Number of distinct user accounts originating from this browser/hardware profile.",
                )
                tatkal_choice = st.selectbox(
                    "Tatkal Quota Window",
                    options=[0, 1],
                    format_func=lambda x: "Active Opening Window (10:00 / 11:00 AM)" if x == 1 else "Standard Booking Period (Off-Peak)",
                    index=0,
                    help="Indicates whether the reservation attempt occurred within the Tatkal opening rush window.",
                )

            with st.expander("Secondary Telemetry & Environmental Attributes", expanded=False):
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    accounts_per_ip = st.number_input(
                        "Accounts per Network IP",
                        min_value=1,
                        max_value=50,
                        value=1,
                        step=1,
                        help="Distinct passenger logins sharing the same egress IP address.",
                    )
                    failed_payments = st.number_input(
                        "Recent Payment Failures",
                        min_value=0,
                        max_value=20,
                        value=0,
                        step=1,
                        help="Failed payment transactions recorded in the preceding 24 hours.",
                    )
                with col_e2:
                    account_age = st.number_input(
                        "Account Age (days)",
                        min_value=0,
                        max_value=2000,
                        value=120,
                        step=10,
                        help="Total days elapsed since passenger account registration.",
                    )
                    click_gap_ms = st.number_input(
                        "Average Click Latency (ms)",
                        min_value=50.0,
                        max_value=10000.0,
                        value=1350.0,
                        step=50.0,
                        help="Mean interval between consecutive DOM interactive mouse/keyboard events.",
                    )

            submitted = st.form_submit_button("EVALUATE SESSION INTEGRITY", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="rail-section-header">SECURITY ASSESSMENT & TELEMETRY</div>', unsafe_allow_html=True)

        if not bundle:
            st.error("No active fraud detector model bundle could be initialized.")
            return

        session_record = {
            "form_fill_seconds": float(form_fill_sec),
            "bookings_last_hour": int(bookings_last_hr),
            "accounts_per_device": int(accounts_per_dev),
            "accounts_per_ip": int(accounts_per_ip),
            "failed_payments": int(failed_payments),
            "account_age_days": int(account_age),
            "is_tatkal_window": int(tatkal_choice),
            "avg_gap_between_clicks_ms": float(click_gap_ms),
        }

        # Auto-evaluate on page load or upon submission
        with st.spinner("Analyzing behavioral session markers..."):
            pred_res = service.predict(session_record, bundle)

        risk_score = pred_res["risk_score"]
        risk_level = pred_res["risk_level"]
        status_light_color = pred_res["status_light"]

        # Signal Light Indicator + Gauge
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem; background-color: #0B1F3A; border: 1px solid #1E3A63; border-radius: 8px; padding: 0.65rem 1rem;">
                <span style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.92rem; font-weight: 700; color: #FFFFFF; letter-spacing: 0.08em; text-transform: uppercase;">
                    THREAT STATUS CLASSIFICATION:
                </span>
                <span>{signal_light(status_light_color, f"{risk_level.upper()} RISK")}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gauge_fig = create_fraud_risk_gauge(risk_score, risk_level)
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Departure board telemetry
        tatkal_label = "TATKAL OPENING WINDOW" if tatkal_choice == 1 else "STANDARD PERIOD"
        board_rows = [
            ("RISK TIER", f"{risk_level.upper()} RISK", status_light_color),
            ("RISK INDEX", f"{risk_score} / 100", None),
            ("FORM SUBMISSION", f"{form_fill_sec:.1f} SECONDS", "red" if form_fill_sec < 4.0 else None),
            ("BOOKING VELOCITY", f"{bookings_last_hr} SESSIONS / HOUR", "red" if bookings_last_hr >= 5 else None),
            ("DEVICE AFFINITY", f"{accounts_per_dev} ACCOUNTS ON DEVICE", "red" if accounts_per_dev >= 4 else None),
            ("WINDOW STATUS", tatkal_label, "amber" if tatkal_choice == 1 else None),
        ]
        departure_board(board_rows)

        # Plain-English reasons card
        reasons_html = "".join([f"<li style='margin-bottom: 0.35rem;'><strong>{r}</strong></li>" for r in pred_res["reasons"]])
        st.markdown(
            f"""
            <div class="yg-ticket" style="margin-top: 1rem;">
                <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #0B1F3A; margin-bottom: 0.4rem;">
                    PRIMARY BEHAVIORAL RISK DRIVERS (EXPLAINABILITY)
                </div>
                <ul style="padding-left: 1.25rem; font-size: 0.85rem; color: #0B1F3A; margin-bottom: 0;">
                    {reasons_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Small Analyst View on the same page
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="rail-section-header">SECURITY ANALYST LOG: RECENT ASSESSED SESSIONS</div>', unsafe_allow_html=True)
    st.caption("Immutable operational audit log of evaluated sessions and automated high-risk security alerts.")

    audit_events = get_events(model_name="fraud_detector", limit=15)
    if audit_events:
        table_rows = []
        for ev in audit_events:
            det = ev.get("details", {})
            score = det.get("risk_score", "N/A")
            level = det.get("risk_level", "HIGH" if ev.get("event_type") == "HIGH_RISK_ALERT" else "N/A")
            reasons_list = det.get("top_reasons", [])
            primary_reason = reasons_list[0] if reasons_list else "Standard automated check"

            table_rows.append({
                "Event ID": ev["id"],
                "Timestamp": ev["timestamp"],
                "Event Type": ev["event_type"],
                "Risk Score": score,
                "Threat Level": str(level).upper(),
                "Primary Driver": primary_reason,
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No session assessments logged yet in this deployment.")

    render_footer()


if __name__ == "__main__":
    main()
