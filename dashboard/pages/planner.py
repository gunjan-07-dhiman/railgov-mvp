"""
dashboard/pages/planner.py
Feature 4: Smart Travel Planner – Simple, user-friendly UI.
Type a travel plan in plain language and get the best matching train itinerary.
"""

from typing import Any, Dict
import pandas as pd
import streamlit as st

from app.base import get_service
import app.modules.planner  # Registers service
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
    page_title=f"{APP_NAME} - Smart Travel Planner",
    page_icon="🚆",
    layout="wide",
)


def main():
    """Render the Smart Travel Planner UI."""
    apply_theme()

    # Sidebar
    render_sidebar_brand()
    render_sidebar_route_navigation(active_page_id="planner")
    render_sidebar_status_strip(active_ver="v1")

    # Header
    render_page_header(
        title="SMART TRAVEL PLANNER",
        subtitle="Tell us where you want to go, your budget, and when you need to arrive — we'll find the best train for you.",
    )

    st.info(
        "💡 **Now Unified!** The Smart Travel Planner is integrated into the **24/7 AI Travel Assistant & Planner**. "
        "You can chat in natural language (English or हिंदी), ask rules, track PNRs, and plan trips all in one place!"
    )
    col_nav1, col_nav2 = st.columns([1.5, 3.5])
    with col_nav1:
        st.page_link("pages/support_chat.py", label="Open 24/7 AI Assistant & Planner →", use_container_width=True)

    service = get_service("journey_planner")

    # ── INPUT SECTION ──────────────────────────────────────────
    st.subheader("✍️ Describe Your Trip")

    # Quick sample buttons
    st.caption("Try a sample plan:")
    s1, s2, s3 = st.columns(3)
    with s1:
        if st.button("Pune → Delhi, under ₹1500, before 10 AM", use_container_width=True):
            st.session_state["planner_query"] = "Pune to Delhi tomorrow under ₹1500, reach before 10 AM"
    with s2:
        if st.button("Mumbai → Goa, Friday night, under ₹900", use_container_width=True):
            st.session_state["planner_query"] = "Mumbai to Goa Friday night under ₹900"
    with s3:
        if st.button("Bangalore → Chennai, morning, under ₹600", use_container_width=True):
            st.session_state["planner_query"] = "Bangalore to Chennai morning express under ₹600"

    default_text = "Pune to Delhi tomorrow under ₹1500, reach before 10 AM"
    active_query = st.session_state.get("planner_query", default_text)

    user_query = st.text_area(
        "Your Travel Plan",
        value=active_query,
        height=80,
        help="Example: 'Mumbai to Goa Friday night under ₹900' or 'Delhi to Varanasi before 2 PM under ₹2000'",
        placeholder="e.g. Mumbai to Goa under ₹1000, reach before 8 PM",
    )

    build_btn = st.button("🔍 FIND BEST TRAIN", type="primary", use_container_width=True)

    # ── RESULTS ──────────────────────────────────────────────────
    if build_btn or active_query != default_text:
        res = service.predict(user_query)
        parsed = res.get("parsed", {})
        best = res.get("best_itinerary")

        st.markdown("---")

        # Show what we understood
        st.subheader("📋 We Understood Your Request")
        u1, u2, u3 = st.columns(3)
        with u1:
            st.info(f"🚂 **Route:** {parsed.get('origin', '?')} → {parsed.get('destination', '?')}")
        with u2:
            st.info(f"💰 **Budget:** ₹{parsed.get('budget', '2000')}")
        with u3:
            deadline = parsed.get('deadline_hr', 24)
            if deadline == 24:
                st.info("🕐 **Arrive by:** Any time")
            else:
                # Convert 24h to readable
                ampm = "AM" if deadline < 12 else "PM"
                display_hr = deadline if deadline <= 12 else deadline - 12
                if display_hr == 0:
                    display_hr = 12
                st.info(f"🕐 **Arrive by:** {display_hr}:00 {ampm}")

        if best:
            # ── BEST MATCH CARD ──
            st.subheader("🏆 Best Train For You")

            with st.container(border=True):
                # Top row: train info + savings
                h1, h2 = st.columns([3, 1])
                with h1:
                    st.markdown(f"### 🚂 {best['train_number']} — {best['train_name']}")
                with h2:
                    if best["within_budget"]:
                        st.success(f"✅ Saves ₹{best['savings']}")
                    else:
                        st.error("❌ Over budget")

                st.divider()

                # Details row
                d1, d2, d3, d4, d5 = st.columns(5)

                with d1:
                    st.caption("FROM")
                    st.markdown(f"**{best['from_station']}**")
                    st.markdown(f"🕐 {best['dep_time']}")

                with d2:
                    st.caption("TO")
                    st.markdown(f"**{best['to_station']}**")
                    st.markdown(f"🕐 {best['arr_time']}")

                with d3:
                    st.caption("DURATION")
                    st.markdown(f"**{best['duration_hrs']} hrs**")

                with d4:
                    st.caption("CLASS & FARE")
                    st.markdown(f"**{best['class']}** — ₹{best['fare']}")

                with d5:
                    st.caption("CLEANLINESS")
                    rating = best['cleanliness']
                    stars = "⭐" * int(rating)
                    st.markdown(f"**{rating}/5.0** {stars}")

            # ── ALTERNATIVE OPTIONS ──
            alts = res.get("alternatives", [])
            if alts:
                st.subheader("🔄 Other Options")
                for alt in alts:
                    with st.container(border=True):
                        a1, a2, a3, a4, a5 = st.columns([2, 1.5, 1, 1.2, 1])
                        with a1:
                            st.markdown(f"**{alt['train_number']} — {alt['train_name']}**")
                        with a2:
                            st.markdown(f"🕐 {alt['dep_time']} → {alt['arr_time']}")
                        with a3:
                            st.markdown(f"⏱️ {alt['duration_hrs']} hrs")
                        with a4:
                            st.markdown(f"💰 **{alt['class']}** — ₹{alt['fare']}")
                        with a5:
                            if alt["within_budget"]:
                                st.markdown("✅ In budget")
                            else:
                                st.markdown("❌ Over budget")
        else:
            st.warning(
                "😕 No trains found matching your request.\n\n"
                "**Try:**\n"
                "- A higher budget\n"
                "- A different route (e.g. Delhi to Mumbai, Pune to Chennai)\n"
                "- A later arrival time"
            )
    else:
        # Placeholder
        st.markdown("---")
        with st.container(border=True):
            st.markdown(
                "#### 👆 Type your travel plan above and click **FIND BEST TRAIN**\n\n"
                "The AI will find trains that match your:\n"
                "- **Route** — where you're going\n"
                "- **Budget** — how much you want to spend\n"
                "- **Arrival time** — when you need to reach\n\n"
                "💡 **Tip:** You can type naturally, like *'Mumbai to Goa under ₹1000, reach before 8 PM'*"
            )

    render_footer()


if __name__ == "__main__":
    main()
