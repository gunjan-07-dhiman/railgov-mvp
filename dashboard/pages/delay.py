"""
dashboard/pages/delay.py
Feature 3: Delay Predictor – Simple, user-friendly UI.
Predicts how late a train might arrive and how crowded the platform will be.
"""

from typing import Any, Dict
import streamlit as st

from app.base import get_service
import app.modules.delay  # Auto-registers delay predictor service
from app.config import APP_NAME
from app.governance.audit import log_event
from app.governance.registry import get_active, register
from dashboard.components.theme import apply_theme
from dashboard.components.ui import (
    render_footer,
    render_page_header,
    render_sidebar_brand,
    render_sidebar_route_navigation,
    render_sidebar_status_strip,
)

st.set_page_config(
    page_title=f"{APP_NAME} - Delay Predictor",
    page_icon="🚆",
    layout="wide",
)


def load_delay_bundle():
    """Load or bootstrap active model bundle for the delay predictor."""
    service = get_service("delay_predictor")
    active = get_active("delay_predictor")
    if not active:
        try:
            base_df = service.load_baseline_df()
            res = service.train(base_df, version="v1")
            register(
                version="v1",
                metrics=res["metrics"],
                training_data_desc="Bootstrap baseline delay model v1",
                status="active",
                model_name="delay_predictor",
            )
            active = get_active("delay_predictor")
        except Exception:
            return None, "Uninitialized"

    try:
        bundle, _ = service.load_model(active["version"])
        return bundle, active["version"]
    except Exception:
        return None, "Uninitialized"


def main():
    """Render the passenger-facing train delay predictor UI."""
    apply_theme()
    bundle, active_ver = load_delay_bundle()
    service = get_service("delay_predictor")

    # Sidebar
    render_sidebar_brand()
    render_sidebar_route_navigation(active_page_id="delay")
    render_sidebar_status_strip(active_ver=active_ver)

    # Page Header
    render_page_header(
        title="DELAY PREDICTOR",
        subtitle="Check if your train will be late and how crowded the station platform will be.",
    )

    # ── INPUT FORM ──────────────────────────────────────────────
    st.subheader("Enter Your Train Details")

    train_options = [
        "12951 - Mumbai Rajdhani Express (Rajdhani)",
        "12301 - Howrah Rajdhani Express (Rajdhani)",
        "12626 - Kerala Express (Superfast)",
        "12417 - Prayagraj Express (Superfast)",
        "14055 - Brahmaputra Mail (Express)",
        "12801 - Purushottam Express (Superfast)",
    ]

    col1, col2 = st.columns(2)

    with col1:
        train_choice = st.selectbox(
            "🚂 Select Your Train",
            train_options,
            index=0,
        )
        train_type = (
            "Rajdhani" if "Rajdhani" in train_choice
            else ("Superfast" if "Superfast" in train_choice else "Express")
        )

        distance_km = st.slider(
            "📏 Distance to Destination (km)",
            min_value=50,
            max_value=2000,
            value=650,
            step=50,
            help="How far is your destination from where you are boarding?",
        )

    with col2:
        departure_hours = {
            0: "12:00 AM (Midnight)", 1: "1:00 AM", 2: "2:00 AM", 3: "3:00 AM",
            4: "4:00 AM", 5: "5:00 AM", 6: "6:00 AM", 7: "7:00 AM",
            8: "8:00 AM", 9: "9:00 AM", 10: "10:00 AM", 11: "11:00 AM",
            12: "12:00 PM (Noon)", 13: "1:00 PM", 14: "2:00 PM", 15: "3:00 PM",
            16: "4:00 PM", 17: "5:00 PM", 18: "6:00 PM", 19: "7:00 PM",
            20: "8:00 PM", 21: "9:00 PM", 22: "10:00 PM", 23: "11:00 PM",
        }
        scheduled_hour = st.selectbox(
            "🕐 Departure Time",
            list(departure_hours.keys()),
            format_func=lambda h: departure_hours[h],
            index=18,
            help="When is your train scheduled to depart?",
        )

        prev_delay = st.slider(
            "⏱️ Delay Already Reported (mins)",
            min_value=0,
            max_value=240,
            value=20,
            step=5,
            help="If the train is already running late from the previous station, enter minutes here.",
        )

    with st.expander("🌧️ Weather & Other Conditions (optional)", expanded=False):
        w1, w2, w3 = st.columns(3)
        with w1:
            rainfall_mm = st.slider(
                "Rainfall (mm)",
                0.0, 150.0, 15.0, 5.0,
                help="Current or forecasted rainfall along the route.",
            )
        with w2:
            dow = st.selectbox(
                "Day of Travel",
                list(range(7)),
                format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d],
                index=3,
            )
        with w3:
            congestion_index = st.slider(
                "Route Congestion",
                0.0, 1.0, 0.55, 0.05,
                help="0 = empty tracks, 1 = very busy route. Leave at default if unsure.",
            )

    st.markdown("")  # small spacer
    predict_clicked = st.button(
        "🔍 CHECK DELAY PREDICTION",
        use_container_width=True,
        type="primary",
    )

    # ── RESULTS ──────────────────────────────────────────────────
    if predict_clicked:
        record: Dict[str, Any] = {
            "distance_km": float(distance_km),
            "scheduled_hour": int(scheduled_hour),
            "day_of_week": int(dow),
            "rainfall_mm": float(rainfall_mm),
            "prev_station_delay_min": float(prev_delay),
            "route_congestion_index": float(congestion_index),
            "train_type": train_type,
        }

        is_valid, issues = service.validate_record(record)
        if not is_valid:
            st.error(f"Please check your input: {', '.join(issues)}")
            log_event(
                "DATA_QUALITY_BLOCK",
                actor="delay_portal",
                details={"record": record, "issues": issues},
                model_name=service.model_name,
            )
        elif not bundle:
            st.warning("The prediction engine is still loading. Please wait a moment and try again.")
        else:
            res = service.predict(record, bundle)

            log_event(
                "PREDICTION",
                actor="delay_portal",
                details={"train": train_choice, "input": record, "prediction": res},
                model_version=active_ver,
                model_name=service.model_name,
            )

            st.markdown("---")
            st.subheader("Prediction Results")

            pred_delay = res["predicted_delay_minutes"]
            range_low = res["range_low"]
            range_high = res["range_high"]
            congestion = res["predicted_congestion"]
            confidence = res["congestion_confidence"]

            # Three clean metric cards
            r1, r2, r3 = st.columns(3)

            with r1:
                with st.container(border=True):
                    if pred_delay < 15:
                        st.markdown("### ✅ On Time")
                    elif pred_delay < 45:
                        st.markdown("### ⚠️ Slight Delay")
                    else:
                        st.markdown("### 🔴 Significant Delay")
                    st.metric(
                        label="Expected Delay",
                        value=f"~{int(round(pred_delay))} min",
                        delta=f"{range_low}–{range_high} min range",
                        delta_color="off",
                    )

            with r2:
                with st.container(border=True):
                    crowd_emoji = "🟢" if congestion == "Low" else ("🟡" if congestion == "Moderate" else "🔴")
                    st.markdown(f"### {crowd_emoji} {congestion} Crowd")
                    st.metric(
                        label="Platform Congestion",
                        value=congestion,
                        delta=f"{confidence * 100:.0f}% confidence",
                        delta_color="off",
                    )

            with r3:
                with st.container(border=True):
                    st.markdown("### 💡 What To Do")
                    if pred_delay >= 30:
                        st.info(f"Arrive **{int(round(pred_delay))} minutes later** than planned. No need to rush to the station.")
                    elif pred_delay >= 10:
                        st.info(f"Minor delay expected. Add **~{int(round(pred_delay))} extra minutes** buffer.")
                    else:
                        st.success("Your train is expected to be on time. Arrive as planned!")

            # Advisory alert
            st.markdown("")
            alert_text = res["proactive_alert"]
            if res["alert_status"] == "DELAY_ALERT":
                st.warning(alert_text)
            else:
                st.success(alert_text)

            # Show delay factors if available (they come from SHAP in the backend)
            top_factors = res.get("top_factors", [])
            if top_factors:
                with st.expander("📊 Why is this delay expected?", expanded=False):
                    for factor in top_factors:
                        st.markdown(f"- {factor['explanation']}")

    else:
        # Placeholder when no prediction yet
        st.markdown("---")
        with st.container(border=True):
            st.markdown(
                "#### 👆 Fill in your train details above and click **CHECK DELAY PREDICTION**\n\n"
                "You'll see:\n"
                "- **Expected delay** in minutes\n"
                "- **Platform crowd level** at the station\n"
                "- **What to do** — practical advice for your journey"
            )

    render_footer()


if __name__ == "__main__":
    main()
