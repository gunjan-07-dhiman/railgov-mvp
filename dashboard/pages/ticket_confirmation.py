"""
dashboard/pages/ticket_confirmation.py
Passenger Portal: Ticket Confirmation Probability Checker.
"""

from typing import Any, Dict, Optional
import streamlit as st

from app.base import get_service
from app.config import VALID_CLASSES, VALID_STATUSES, VALID_TRAIN_TYPES
from app.governance.audit import log_event
from dashboard.components.theme import render_header


def render_ticket_confirmation_page(bundle: Optional[Any], active_ver: str) -> None:
    """Render the passenger-facing ticket confirmation predictor UI."""
    render_header(
        title="Indian Railways Ticket Confirmation Predictor",
        subtitle="Check whether your Waitlisted (WL) or RAC ticket will get confirmed before chart preparation.",
        icon="🎫",
    )

    col_form, _, col_result = st.columns([1.2, 0.1, 1.2])

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
                format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
                index=4,
            )
            festival_rush = st.checkbox("Festival / Holiday Rush Period", value=False)
            route_pop = st.slider("Route Demand Intensity", 0.0, 1.0, 0.60, 0.05)

        predict_button = st.button("🔮 Check Confirmation Chance", use_container_width=True, type="primary")

    with col_result:
        st.markdown("### 📊 Prediction Result")

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

            service = get_service("ticket_predictor")
            is_valid, issues = service.validate_record(record)
            if not is_valid:
                st.error(f"Please check your input: {', '.join(issues)}")
                log_event(
                    "DATA_QUALITY_BLOCK",
                    actor="passenger_portal",
                    details={"record": record, "issues": issues},
                    model_name=service.model_name,
                )
            elif not bundle:
                st.error("Prediction engine is currently initializing. Please retry in a few moments.")
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
                    st.success("### 🟢 HIGH CHANCE OF CONFIRMATION")
                    status_desc = "Your ticket has a high likelihood of getting confirmed before chart preparation."
                elif status_str == "Uncertain":
                    st.warning("### 🟡 MODERATE / UNCERTAIN CHANCE")
                    status_desc = "Confirmation depends on cancellations in the final 24-48 hours before departure."
                else:
                    st.error("### 🔴 LOW CHANCE OF CONFIRMATION")
                    status_desc = "Unlikely to confirm. We recommend booking Tatkal or choosing an alternate train."

                st.markdown(f"**Confirmation Probability:** `{prob_pct:.1f}%` ({res['confidence']} Confidence)")
                st.info(status_desc)

                st.markdown("#### 💡 Key Deciding Factors")
                for factor in res.get("top_factors", []):
                    st.markdown(f"- **{factor['explanation']}**")
        else:
            st.info("👈 Enter your ticket details on the left and click **'Check Confirmation Chance'**.")
