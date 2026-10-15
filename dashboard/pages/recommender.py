"""
dashboard/pages/recommender.py
Feature 2: Train Recommender Page for YatraGov AI.
Suggests optimal trains based on passenger priorities (fastest, cheapest, cleanest, balanced).
Supports custom station input, individual clean cards, and side-by-side train comparison.
"""

from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from app.base import get_service
import app.modules.recommender  # Registers service
from app.config import APP_NAME
from dashboard.components.theme import apply_theme
from dashboard.components.ui import (
    render_footer,
    render_page_header,
    render_sidebar_brand,
    render_sidebar_route_navigation,
    render_sidebar_status_strip,
)

st.set_page_config(
    page_title=f"{APP_NAME} - Train Recommender",
    page_icon="🚆",
    layout="wide",
)


def main():
    """Render the Train Recommender UI."""
    apply_theme()

    # Dark navy sidebar
    render_sidebar_brand()
    render_sidebar_route_navigation(active_page_id="recommender")
    render_sidebar_status_strip(active_ver="v1")

    # Header
    render_page_header(
        title="TRAIN RECOMMENDER",
        subtitle="Suggests the best trains based on what you actually care about (fastest journey, cheapest fare, cleanest train).",
    )

    service = get_service("train_recommender")

    # Search Bar & Parameters
    st.markdown('<div class="rail-section-header">SEARCH CORRIDOR & YOUR PREFERENCES</div>', unsafe_allow_html=True)

    # Quick Popular Route Chips
    st.caption("Popular corridors (click to auto-fill or enter custom stations below):")
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        if st.button("Delhi → Mumbai", key="rt_dl_mum", use_container_width=True):
            st.session_state["from_stn_input"] = "New Delhi"
            st.session_state["to_stn_input"] = "Mumbai Central"
    with p2:
        if st.button("Delhi → Pune", key="rt_dl_pune", use_container_width=True):
            st.session_state["from_stn_input"] = "Delhi"
            st.session_state["to_stn_input"] = "Pune"
    with p3:
        if st.button("Mumbai → Goa", key="rt_mum_goa", use_container_width=True):
            st.session_state["from_stn_input"] = "Mumbai Central"
            st.session_state["to_stn_input"] = "Madgaon (Goa)"
    with p4:
        if st.button("Delhi → Varanasi", key="rt_dl_vns", use_container_width=True):
            st.session_state["from_stn_input"] = "New Delhi"
            st.session_state["to_stn_input"] = "Varanasi"
    with p5:
        if st.button("Bangalore → Chennai", key="rt_blr_chn", use_container_width=True):
            st.session_state["from_stn_input"] = "Bangalore"
            st.session_state["to_stn_input"] = "Chennai Central"

    # Input Fields
    col_from, col_to, col_priority, col_class = st.columns([1.2, 1.2, 1.4, 0.9])

    with col_from:
        from_stn = st.text_input(
            "From Station / Code",
            value=st.session_state.get("from_stn_input", "New Delhi"),
            help="Type any station or code (e.g. NDLS, Mumbai, Pune, Jaipur, Lucknow, Patna...)",
        )
    with col_to:
        to_stn = st.text_input(
            "To Station / Code",
            value=st.session_state.get("to_stn_input", "Mumbai Central"),
            help="Type any destination or code (e.g. MMCT, Pune, Delhi, Goa, Varanasi, Chennai...)",
        )
    with col_priority:
        priority_options = {
            "fastest": "Fastest Journey (Shortest time)",
            "cheapest": "Cheapest Fare (Budget savings)",
            "cleanest": "Cleanest & Most Punctual",
            "balanced": "Best Balanced Choice",
        }
        priority = st.selectbox(
            "What do you care about most?",
            list(priority_options.keys()),
            format_func=lambda k: priority_options[k],
            index=0,
        )
    with col_class:
        travel_class = st.selectbox(
            "Travel Class",
            ["SL", "3A", "2A", "1A"],
            format_func=lambda c: {"SL": "Sleeper (SL)", "3A": "AC 3 Tier (3A)", "2A": "AC 2 Tier (2A)", "1A": "First AC (1A)"}.get(c, c),
            index=0,
        )

    st.markdown('<div style="margin-top: 0.25rem;"></div>', unsafe_allow_html=True)
    search_clicked = st.button("RECOMMEND BEST TRAINS", type="primary", use_container_width=True)

    st.markdown('<div style="margin-top: 1.25rem;"></div>', unsafe_allow_html=True)

    # Fetch ranked train recommendations
    res = service.predict({
        "from_station": from_stn,
        "to_station": to_stn,
        "priority": priority,
        "travel_class": travel_class,
    })

    recs: List[Dict[str, Any]] = res.get("recommendations", [])
    resolved_from = res.get("from_station", from_stn)
    resolved_to = res.get("to_station", to_stn)

    # Header showing summary
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 1.25rem; font-weight: 700; color: #0B1F3A; text-transform: uppercase;">
                FOUND {len(recs)} TRAINS: {resolved_from} → {resolved_to}
            </div>
            <div style="font-size: 0.8rem; color: #5B6B82;">
                Preference: <strong>{priority_options[priority]}</strong> | Class: <strong>{travel_class}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if recs:
        # Tabbed view: Individual Cards vs Side-by-Side Comparison
        tab_cards, tab_compare = st.tabs(["Individual Train Cards", "Side-by-Side Comparison Matrix"])

        # TAB 1: INDIVIDUAL TRAIN CARDS
        with tab_cards:
            for idx, train in enumerate(recs):
                is_top = idx == 0

                # Render with native Streamlit container to guarantee 100% clean rendering
                with st.container(border=True):
                    head_col1, head_col2 = st.columns([3, 1.2])

                    with head_col1:
                        rank_tag = "★ #1 TOP AI RECOMMENDATION" if is_top else f"#{train['rank']} ALTERNATIVE"
                        tag_color = "#059669" if is_top else "#5B6B82"
                        st.markdown(
                            f"<span style='color: {tag_color}; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em;'>{rank_tag}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"#### {train['train_number']} — {train['train_name']} &nbsp;`{train['train_type']}`")
                        st.caption(f"**Why recommended**: {train['ai_reason']}")

                    with head_col2:
                        st.metric("AI Match", f"{train['match_score']}%", "Top Match" if is_top else None)

                    st.divider()

                    # 4 Metrics Columns
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.caption("DEPARTURE → ARRIVAL")
                        st.markdown(f"**`{train['dep_time']}` → `{train['arr_time']}`**")
                        st.caption(f"{resolved_from} to {resolved_to}")
                    with m2:
                        st.caption("JOURNEY DURATION")
                        st.markdown(f"**{train['duration_hrs']} hrs**")
                    with m3:
                        st.caption(f"BASE FARE ({travel_class})")
                        st.markdown(f"<span style='font-size: 1.2rem; font-weight: 700; color: #D97706;'>₹{train['fare']}</span>", unsafe_allow_html=True)
                    with m4:
                        st.caption("CLEANLINESS & ON-TIME")
                        st.markdown(f"**★ {train['cleanliness_rating']} / 5.0**")
                        st.caption(f"{train['punctuality_pct']}% punctuality record")

        # TAB 2: SIDE-BY-SIDE COMPARISON MATRIX
        with tab_compare:
            st.markdown("##### Head-to-Head Comparison Across All Matching Trains")
            compare_data = []
            for t in recs:
                compare_data.append({
                    "Rank": f"#{t['rank']}",
                    "Train": f"{t['train_number']} {t['train_name']}",
                    "Type": t["train_type"],
                    "Dep → Arr": f"{t['dep_time']} - {t['arr_time']}",
                    "Duration": f"{t['duration_hrs']} hrs",
                    f"Fare ({travel_class})": f"₹{t['fare']}",
                    "Cleanliness": f"★ {t['cleanliness_rating']}/5.0",
                    "Punctuality": f"{t['punctuality_pct']}%",
                    "AI Match Score": f"{t['match_score']}%",
                })
            st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

            st.caption("Comparison highlights: Rajdhani & Shatabdi trains offer highest speed and cleanliness; Superfast and Express services offer maximum budget economy.")

    else:
        st.info("No trains found for this route. Try typing different station names or use the quick corridor buttons above.")

    # Data Source & Real-Time API Notice
    with st.expander("About Real-Time Train Data & RapidAPI Integration", expanded=False):
        st.markdown(
            """
            **How Train Data Works in YatraGov AI:**
            - **Real-Time PNR Tracking**: The 24/7 Chatbot (Feature 6) uses your live RapidAPI key from `.env` to fetch live PNR and booking status directly from Indian Railways PRS servers.
            - **Train Timetable & Tariff Engine**: Powered by our verified Indian Railways corridor database (`data/trains.csv`) with official class tariffs, train numbers, and timetables across India.
            - **Custom Station Resolution**: If you type any two stations across India not yet in the static catalog, our dynamic corridor engine computes the realistic rail distance, transit time, and IRCTC fare structure.
            - **Adding Live Train Search API**: If you have a RapidAPI subscription with endpoints for `/trainBetweenStations`, you can configure the host in `.env` to fetch live real-time berth availability.
            """
        )

    render_footer()


if __name__ == "__main__":
    main()
