"""
dashboard/pages/fare.py
Passenger Dynamic Fare Movement & Demand Surge Advisor for YatraGov AI.
Theme: "Departure Board & Ticket" (Indian Railway Signage & Warm Paper).
"""

from typing import Any, Dict
import streamlit as st

from app.base import get_service
from app.config import APP_NAME
from app.governance.audit import log_event
from dashboard.components.charts import create_fare_horizon_chart
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
    page_title=f"{APP_NAME} - Fare & Demand Surge",
    page_icon="🚆",
    layout="wide",
)


def main():
    """Render the Passenger Dynamic Fare Movement and Best Booking Window UI."""
    apply_theme()

    # Dark navy sidebar
    render_sidebar_brand()
    render_sidebar_route_navigation(active_page_id="fare")

    service = get_service("fare_predictor")
    bundle, meta = load_active_bundle("fare_predictor")
    active_ver = meta.get("version", "v1") if isinstance(meta, dict) else "v1"
    render_sidebar_status_strip(active_ver=active_ver)

    # Page Header in Barlow Condensed with saffron underline
    render_page_header(
        title="DYNAMIC FARE PREDICTOR",
        subtitle="Predicts when ticket prices or Tatkal demand will surge and identifies the best purchase window.",
    )

    col_form, col_space, col_result = st.columns([1.15, 0.05, 1.15])

    with col_form:
        st.markdown('<div class="rail-section-header">JOURNEY & BOOKING PARAMETERS</div>', unsafe_allow_html=True)

        class_options = ["SL", "3A", "2A", "1A", "CC"]
        class_labels = {
            "SL": "Sleeper Class (SL)",
            "3A": "AC 3 Tier (3A)",
            "2A": "AC 2 Tier (2A)",
            "1A": "AC First Class (1A)",
            "CC": "AC Chair Car (CC)",
        }
        travel_class = st.selectbox(
            "Travel Class",
            class_options,
            format_func=lambda c: class_labels.get(c, c),
            index=1,
            help="Select reserved accommodation class.",
        )

        train_options = ["Express", "Superfast", "Rajdhani"]
        train_type = st.selectbox(
            "Train Category",
            train_options,
            index=1,
            help="Category determines priority surcharges and dynamic pricing scales.",
        )

        distance_km = st.slider(
            "Journey Distance (km)",
            min_value=50,
            max_value=2500,
            value=750,
            step=25,
            help="Total route distance in kilometers.",
        )

        days_before = st.slider(
            "Days Before Journey (Booking Lead Time)",
            min_value=0,
            max_value=120,
            value=14,
            step=1,
            help="Number of days remaining until the train departs (0 = departure day).",
        )

        is_festival = st.toggle(
            "Peak Festival Rush Period (Diwali / Chhath / Puja)",
            value=False,
            help="Enable if journey occurs during high-demand seasonal holiday clusters.",
        )

        with st.expander("Optional Route Details & Quoted Fare Check", expanded=False):
            route_popularity = st.slider(
                "Route Passenger Demand Index",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.05,
                help="Historical occupancy density on this corridor (0.0 = low traffic, 1.0 = heavy rush).",
            )
            seats_pct = st.slider(
                "Remaining Available Seats (%)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
                help="Estimated current seat inventory remaining on this service.",
            )
            month = st.selectbox(
                "Travel Month",
                list(range(1, 13)),
                format_func=lambda m: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1],
                index=9,
            )
            day_of_week = st.selectbox(
                "Day of Week",
                list(range(7)),
                format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
                index=4,
            )
            quoted_fare = st.number_input(
                "Fare You Were Quoted (INR, Optional)",
                min_value=0,
                max_value=25000,
                value=0,
                step=50,
                help="Enter any quoted ticket price to run a governed price anomaly check.",
            )

        submit_btn = st.button("ANALYZE FARE & DEMAND SURGE", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<div class="rail-section-header">FARE TELEMETRY & BOOKING ADVICE</div>', unsafe_allow_html=True)

        if submit_btn:
            record: Dict[str, Any] = {
                "travel_class": travel_class,
                "train_type": train_type,
                "distance_km": float(distance_km),
                "days_before_journey": int(days_before),
                "day_of_week": int(day_of_week),
                "month": int(month),
                "is_festival_season": 1 if is_festival else 0,
                "route_popularity": float(route_popularity),
                "seats_available_pct": float(seats_pct),
            }

            is_valid, issues = service.validate_record(record)
            if not is_valid:
                st.error(f"Input validation error: {', '.join(issues)}")
                log_event(
                    "DATA_QUALITY_BLOCK",
                    actor="fare_portal",
                    details={"record": record, "issues": issues},
                    model_name=service.model_name,
                )
            elif not bundle:
                st.error("Fare prediction engine is initializing. Please retry in a moment.")
            else:
                pred_res = service.predict(record, bundle)
                best_res = service.best_booking_window(record, bundle)

                log_event(
                    "PREDICTION",
                    actor="fare_portal",
                    details={"input": record, "prediction": pred_res, "booking_window": best_res},
                    model_version=active_ver,
                    model_name=service.model_name,
                )

                # Anomaly check if quoted fare provided
                anomaly_res = None
                if quoted_fare > 0:
                    anomaly_res = service.check_price_anomaly(float(quoted_fare), float(pred_res["predicted_fare_inr"]))
                    if anomaly_res.get("is_anomaly"):
                        log_event(
                            "PRICE_ANOMALY",
                            actor="fare_portal",
                            details={"quoted_fare": quoted_fare, "predicted_fare": pred_res["predicted_fare_inr"], "anomaly": anomaly_res},
                            model_version=active_ver,
                            model_name=service.model_name,
                        )

                # Departure board telemetry
                demand_val = pred_res["predicted_demand"]
                demand_status = "red" if demand_val == "Surge" else ("green" if demand_val == "Normal" else None)
                savings_str = f"₹{best_res['estimated_saving']:,}" if best_res["estimated_saving"] > 0 else "Best current rate"

                board_rows = [
                    ("PREDICTED FARE", f"₹{pred_res['predicted_fare_inr']:,}", None),
                    ("LIKELY RANGE", f"₹{pred_res['range_low']:,} – ₹{pred_res['range_high']:,}", None),
                    ("DEMAND LEVEL", f"{demand_val.upper()} DEMAND ({pred_res['demand_confidence']*100:.0f}% CONF)", demand_status),
                    ("BEST DAY TO BOOK", f"Day {best_res['best_day']} ahead" + (" (Today is optimal)" if best_res['best_day'] == days_before else ""), "green"),
                    ("YOU COULD SAVE", savings_str, "green" if best_res["estimated_saving"] > 0 else None),
                ]
                departure_board(board_rows)

                # Horizon Trajectory Line Chart
                horizon_fig = create_fare_horizon_chart(
                    days_series=best_res["days_series"],
                    fares_series=best_res["fares_series"],
                    current_days=days_before,
                    best_day=best_res["best_day"],
                    best_fare=best_res["best_fare"],
                    today_fare=best_res["today_fare"],
                )
                st.plotly_chart(horizon_fig, use_container_width=True)

                # Deciding factors ticket stub
                factors_html = "".join([f"<li style='margin-bottom: 0.35rem;'><strong>{f['explanation']}</strong></li>" for f in pred_res.get("top_factors", [])])
                st.markdown(
                    f"""
                    <div class="yg-ticket" style="margin-top: 1rem;">
                        <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #0B1F3A; margin-bottom: 0.4rem;">
                            PRIMARY FARE DRIVERS (TREESHAP EXPLAINABILITY)
                        </div>
                        <ul style="padding-left: 1.25rem; font-size: 0.85rem; color: #0B1F3A; margin-bottom: 0;">
                            {factors_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Quoted price anomaly card
                if anomaly_res:
                    is_anom = anomaly_res["is_anomaly"]
                    anom_status = "caution" if is_anom else "normal"
                    anom_tag = "UNUSUAL QUOTE" if is_anom else "STANDARD QUOTE"
                    border_color = "#EF4444" if is_anom else "#22C55E"

                    st.markdown(
                        f"""
                        <div class="yg-ticket" style="margin-top: 1rem; border-left: 4px solid {border_color};">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                                <span style="font-family: 'Barlow Condensed', sans-serif; font-size: 0.88rem; font-weight: 700; text-transform: uppercase; color: #0B1F3A;">
                                    PRICE VERIFICATION CHECK
                                </span>
                                {signal_light(anom_status, anom_tag)}
                            </div>
                            <div style="font-size: 0.85rem; color: #5B6B82;">
                                {anomaly_res['message']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Enter journey parameters on the left and click Analyze to view fare projections, demand surges, and optimal booking horizon.")

        st.markdown(
            """
            <div style="font-size: 0.76rem; color: #5B6B82; margin-top: 1.5rem; text-align: center;">
                Demo data: Fare movements and demand surge estimates are generated from academic synthetic patterns. Not official IRCTC tariffs.
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_footer()


if __name__ == "__main__":
    main()
