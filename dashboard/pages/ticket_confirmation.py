"""
dashboard/pages/ticket_confirmation.py
Passenger Portal: Ticket Confirmation Probability Checker for YatraGov AI.
Theme: "Departure Board & Ticket".
"""

from typing import Any, Dict
import streamlit as st

from app.base import get_service
import app.modules.ticket_predictor  # Auto-registers ticket predictor service
from app.config import APP_NAME, VALID_CLASSES, VALID_STATUSES, VALID_TRAIN_TYPES
from app.governance.audit import log_event
from app.governance.registry import get_active
from dashboard.components.theme import apply_theme
from dashboard.components.ui import (
    render_footer,
    render_page_header,
    render_sidebar_brand,
    render_sidebar_route_navigation,
    render_sidebar_status_strip,
    ticket_stub,
)

st.set_page_config(
    page_title=f"{APP_NAME} - Ticket Confirmation",
    page_icon="🚆",
    layout="wide",
)


def main():
    """Render the passenger-facing ticket confirmation predictor UI."""
    apply_theme()

    service = get_service("ticket_predictor")
    active_info = get_active("ticket_predictor")
    active_ver = active_info["version"] if active_info else "v1"

    try:
        bundle, _ = service.load_model(active_ver)
    except Exception:
        bundle = None

    # Sidebar layout
    render_sidebar_brand()
    render_sidebar_route_navigation(active_page_id="ticket_confirmation")
    render_sidebar_status_strip(active_ver=active_ver)

    # Page Header in Barlow Condensed with saffron underline
    render_page_header(
        title="TICKET CONFIRM PREDICTOR",
        subtitle="Calibrated Waitlist & RAC confirmation odds (e.g. 'Your WL 12 has an 82% chance of confirming').",
    )

    col_form, col_space, col_result = st.columns([1.15, 0.05, 1.15])

    with col_form:
        st.markdown('<div class="rail-section-header">JOURNEY PARAMETERS</div>', unsafe_allow_html=True)

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

        with st.expander("Additional Route & Season Details", expanded=False):
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
                format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
                index=4,
            )
            festival_rush = st.checkbox("Festival / Holiday Rush Period", value=False)
            route_pop = st.slider("Route Demand Intensity", 0.0, 1.0, 0.60, 0.05)

        predict_button = st.button("CHECK CONFIRMATION CHANCE", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<div class="rail-section-header">CONFIRMATION FORECAST</div>', unsafe_allow_html=True)

        if predict_button:
            record: Dict[str, Any] = {
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

            is_valid, issues = service.validate_record(record)
            if not is_valid:
                st.error(f"Input validation error: {', '.join(issues)}")
                log_event(
                    "DATA_QUALITY_BLOCK",
                    actor="passenger_portal",
                    details={"record": record, "issues": issues},
                    model_name=service.model_name,
                )
            elif not bundle:
                st.error("Prediction engine is currently initializing. Please retry in a moment.")
            else:
                res = service.predict(record, bundle)
                prob_pct = res["probability"] * 100.0
                status_str = res["predicted_status"]

                log_event(
                    "PREDICTION",
                    actor="passenger_portal",
                    details={"ticket": record, "prediction": res},
                    model_version=active_ver,
                    model_name=service.model_name,
                )

                if status_str == "Likely Confirmed":
                    status_desc = "High probability of confirmation before chart preparation."
                elif status_str == "Uncertain":
                    status_desc = "Moderate odds. Confirmation depends on last-minute cancellations."
                else:
                    status_desc = "Low confirmation likelihood. Recommend booking Tatkal or alternate options."

                # Signature ticket stub with half-gauge and TreeSHAP factors
                ticket_stub(
                    status_str=status_str,
                    prob_pct=prob_pct,
                    details={"desc": status_desc, "confidence": res.get("confidence")},
                    top_factors=res.get("top_factors", []),
                )
        else:
            st.markdown(
                """
                <div class="yg-ticket" style="text-align: center; padding: 3rem 1.5rem; color: #5B6B82;">
                    <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 1.2rem; font-weight: 700; color: #0B1F3A; text-transform: uppercase;">
                        TICKET STUB STANDBY
                    </div>
                    <div style="font-size: 0.88rem; margin-top: 0.35rem; max-width: 380px; margin-left: auto; margin-right: auto;">
                        Fill in your journey details on the left and press <strong>'CHECK CONFIRMATION CHANCE'</strong> to generate your predictive ticket stub.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_footer()


if __name__ == "__main__":
    main()
