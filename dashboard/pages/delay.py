"""
dashboard/pages/delay.py
Passenger Portal: Train Delay Minutes Forecasting & Platform Congestion for YatraGov AI.
Theme: "Departure Board & Ticket".
"""

from typing import Any, Dict
import streamlit as st

from app.base import get_service
import app.modules.delay  # Auto-registers delay predictor service
from app.config import APP_NAME, VALID_TRAIN_TYPES
from app.governance.audit import log_event
from app.governance.registry import get_active, register
from dashboard.components.theme import apply_theme
from dashboard.components.ui import (
    departure_board,
    render_footer,
    render_page_header,
    render_sidebar_brand,
    render_sidebar_route_navigation,
    render_sidebar_status_strip,
)

st.set_page_config(
    page_title=f"{APP_NAME} - Train Delay & Platform Congestion",
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
    """Render the passenger-facing train delay and platform congestion UI."""
    apply_theme()
    bundle, active_ver = load_delay_bundle()
    service = get_service("delay_predictor")

    # Dark navy sidebar
    render_sidebar_brand()
    render_sidebar_route_navigation(active_page_id="delay")
    render_sidebar_status_strip(active_ver=active_ver)

    # Page Header in Barlow Condensed with saffron underline
    render_page_header(
        title="TRAIN DELAY & PLATFORM CONGESTION",
        subtitle="Expected arrival error margin range, station platform crowd levels, and proactive arrival guidance.",
    )

    col_form, col_space, col_result = st.columns([1.15, 0.05, 1.15])

    with col_form:
        st.markdown('<div class="rail-section-header">JOURNEY & ROUTE TELEMETRY</div>', unsafe_allow_html=True)

        train_options = [
            "12951 - Mumbai Rajdhani Express (Rajdhani)",
            "12301 - Howrah Rajdhani Express (Rajdhani)",
            "12626 - Kerala Express (Superfast)",
            "12417 - Prayagraj Express (Superfast)",
            "14055 - Brahmaputra Mail (Express)",
            "12801 - Purushottam Express (Superfast)",
        ]
        train_choice = st.selectbox("Select Scheduled Train", train_options, index=0)
        train_type = "Rajdhani" if "Rajdhani" in train_choice else ("Superfast" if "Superfast" in train_choice else "Express")

        distance_km = st.slider(
            "Remaining Journey Distance (km)",
            min_value=50,
            max_value=2000,
            value=650,
            step=25,
            help="Total kilometers between current boarding point and destination.",
        )

        scheduled_hour = st.slider(
            "Scheduled Departure Hour",
            min_value=0,
            max_value=23,
            value=18,
            format="%02d:00",
            help="Hour of departure (24-hour clock).",
        )

        prev_delay = st.number_input(
            "Upstream Delay at Preceding Junction (mins)",
            min_value=0,
            max_value=240,
            value=20,
            help="How many minutes late the train departed the preceding junction.",
        )

        with st.expander("Additional Route & Weather Conditions", expanded=False):
            rainfall_mm = st.slider("Route Rainfall Intensity (mm)", 0.0, 150.0, 15.0, 2.5)
            dow = st.selectbox(
                "Journey Day of Week",
                list(range(7)),
                format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
                index=3,
            )
            congestion_index = st.slider("Corridor Track Congestion Density", 0.0, 1.0, 0.55, 0.05)

        forecast_button = st.button("FORECAST DELAY & STATION STATUS", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<div class="rail-section-header">DEPARTURE BOARD & FORECAST</div>', unsafe_allow_html=True)

        if forecast_button:
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
                st.error(f"Input validation error: {', '.join(issues)}")
                log_event(
                    "DATA_QUALITY_BLOCK",
                    actor="delay_portal",
                    details={"record": record, "issues": issues},
                    model_name=service.model_name,
                )
            elif not bundle:
                st.error("Delay prediction engine is initializing. Please retry in a moment.")
            else:
                res = service.predict(record, bundle)

                log_event(
                    "PREDICTION",
                    actor="delay_portal",
                    details={"train": train_choice, "input": record, "prediction": res},
                    model_version=active_ver,
                    model_name=service.model_name,
                )

                # Signature Departure Board (Navy panel with amber Share Tech Mono text)
                pred_delay = res["predicted_delay_minutes"]
                delay_status = "green" if pred_delay < 15 else ("red" if pred_delay >= 45 else None)

                board_rows = [
                    ("EXPECTED ARRIVAL", res["delay_range_display"], delay_status),
                    ("PLATFORM CROWD", f"{res['predicted_congestion'].upper()} CROWD ({res['congestion_confidence']*100:.0f}% CONF)", None),
                    ("ADVISORY", res["proactive_alert"], None),
                ]
                departure_board(board_rows)

                # Explanatory factors ticket stub
                factors_html = "".join([f"<li style='margin-bottom: 0.35rem;'><strong>{f['explanation']}</strong></li>" for f in res.get("top_factors", [])])
                st.markdown(
                    f"""
                    <div class="yg-ticket" style="margin-top: 1rem;">
                        <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.88rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #0B1F3A; margin-bottom: 0.4rem;">
                            PRIMARY DELAY DRIVERS (TREESHAP EXPLANATION)
                        </div>
                        <ul style="padding-left: 1.25rem; font-size: 0.85rem; color: #0B1F3A; margin-bottom: 0;">
                            {factors_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="yg-departure-board" style="text-align: center; padding: 2.75rem 1.5rem; opacity: 0.85;">
                    <div style="font-family: 'Share Tech Mono', monospace; font-size: 1.15rem; color: #FF9F1C;">
                        --- DEPARTURE TELEMETRY OFFLINE ---
                    </div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.45rem;">
                        Select scheduled train and route parameters on the left and press <strong>'FORECAST DELAY & STATION STATUS'</strong> to populate the live departure board.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_footer()


if __name__ == "__main__":
    main()
