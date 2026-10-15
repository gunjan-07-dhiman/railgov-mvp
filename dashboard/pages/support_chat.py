"""
dashboard/pages/support_chat.py
24/7 AI Travel Assistant & Planner for YatraGov AI.
Unified conversational railway guide:
- Natural language trip planning with constraint-satisfaction itineraries
- Real-time RapidAPI & verified PNR status tracking
- Multilingual extractive grounding over verified IRCTC policy documentation
"""

import os
from typing import Any, Dict, List
import streamlit as st

from app.base import get_service
from app.config import APP_NAME
from dashboard.components.theme import apply_theme
from dashboard.components.ui import (
    load_active_bundle,
    render_footer,
    render_page_header,
    render_sidebar_brand,
    render_sidebar_route_navigation,
    render_sidebar_status_strip,
)

st.set_page_config(
    page_title=f"{APP_NAME} - 24/7 AI Travel Assistant & Planner",
    page_icon="🚆",
    layout="wide",
)


def _render_trip_card(trip_data: Dict[str, Any]):
    """Render high-aesthetic interactive train itinerary card."""
    best = trip_data.get("best_itinerary")
    if not best:
        return

    parsed = trip_data.get("parsed", {})
    budget_cap = parsed.get("budget", 2000)

    # Best match hero container
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div style="font-size: 0.74rem; font-weight: 700; color: #10B981; letter-spacing: 0.08em; text-transform: uppercase;">
                    🏆 RECOMMENDED ITINERARY
                </div>
                <div style="font-size: 0.76rem; font-weight: 600; background: {'rgba(16, 185, 129, 0.12)' if best['within_budget'] else 'rgba(239, 68, 68, 0.12)'}; color: {'#059669' if best['within_budget'] else '#DC2626'}; padding: 2px 10px; border-radius: 12px; border: 1px solid {'#10B981' if best['within_budget'] else '#EF4444'}40;">
                    {'✅ Within Budget • Saves ₹' + str(best['savings']) if best['within_budget'] else '⚠️ Exceeds Budget (Cap ₹' + str(budget_cap) + ')'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<h3 style='margin: 0; color: #0B1F3A;'>🚂 {best['train_number']} — {best['train_name']}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption("DEPARTURE")
            st.markdown(f"**{best['dep_time']}**")
            st.markdown(f"<span style='color: #5B6B82; font-size: 0.85rem;'>{best['from_station']}</span>", unsafe_allow_html=True)

        with c2:
            st.caption("ARRIVAL")
            st.markdown(f"**{best['arr_time']}**")
            st.markdown(f"<span style='color: #5B6B82; font-size: 0.85rem;'>{best['to_station']}</span>", unsafe_allow_html=True)

        with c3:
            st.caption("DURATION & CLASS")
            st.markdown(f"**{best['duration_hrs']} hrs**")
            st.markdown(f"<span style='color: #0B1F3A; font-weight: 600; font-size: 0.85rem;'>Class: {best['class']}</span>", unsafe_allow_html=True)

        with c4:
            st.caption("FARE & CLEANLINESS")
            st.markdown(f"**₹{best['fare']}**")
            rating = best.get("cleanliness", 4.5)
            st.markdown(f"<span style='color: #D97706; font-size: 0.82rem;'>{'⭐' * int(rating)} {rating}/5.0</span>", unsafe_allow_html=True)

    # Alternatives expander
    alts = trip_data.get("alternatives", [])
    if alts:
        with st.expander(f"🔄 View {len(alts)} Other Train Options on this Route"):
            for alt in alts:
                with st.container(border=True):
                    a_col1, a_col2, a_col3 = st.columns([2.5, 2.0, 1.5])
                    with a_col1:
                        st.markdown(f"**{alt['train_number']} — {alt['train_name']}**")
                        st.caption(f"Route: {alt['from_station']} → {alt['to_station']}")
                    with a_col2:
                        st.markdown(f"🕐 **{alt['dep_time']} → {alt['arr_time']}** ({alt['duration_hrs']}h)")
                        st.caption(f"Cleanliness: ⭐ {alt.get('cleanliness', 4.0)}/5.0")
                    with a_col3:
                        budget_badge = "✅ Within budget" if alt["within_budget"] else "❌ Over budget"
                        st.markdown(f"**₹{alt['fare']}** ({alt['class']})")
                        st.caption(budget_badge)


def _render_pnr_card(pnr_data: Dict[str, Any]):
    """Render clean, official-style PNR reservation status card."""
    with st.container(border=True):
        st_color = "#059669" if "CNF" in str(pnr_data.get("current_status", "")) else "#D97706"
        is_live = pnr_data.get("is_live", False)
        source_badge = "⚡ RapidAPI Real-Time" if is_live else "💾 Verified Database"

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div style="font-size: 0.74rem; font-weight: 700; color: #0B1F3A; letter-spacing: 0.08em; text-transform: uppercase;">
                    INDIAN RAILWAYS PASSENGER RESERVATION STATUS
                </div>
                <div style="font-size: 0.72rem; font-weight: 600; background: rgba(11, 31, 58, 0.08); color: #0B1F3A; padding: 2px 8px; border-radius: 10px;">
                    {source_badge}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.caption("PNR NUMBER")
            st.markdown(f"**{pnr_data.get('pnr_number', 'N/A')}**")
            st.caption(f"Passenger: {pnr_data.get('passenger_name', 'Passenger')}")
        with p2:
            st.caption("TRAIN")
            st.markdown(f"**{pnr_data.get('train_number', 'N/A')}**")
            st.caption(f"{pnr_data.get('train_name', 'Express')}")
        with p3:
            st.caption("DATE & CLASS")
            st.markdown(f"**{pnr_data.get('journey_date', 'N/A')}**")
            st.caption(f"Class: {pnr_data.get('travel_class', 'SL')}")
        with p4:
            st.caption("CURRENT STATUS")
            curr_st = pnr_data.get("current_status", "CNF")
            st.markdown(
                f"<span style='font-size: 1.15rem; font-weight: 800; color: {st_color};'>{curr_st}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Chart: {pnr_data.get('chart_status', 'Chart Not Prepared')}")


def main():
    """Render the Passenger Multilingual Grounded Support Chatbot & Trip Planner UI."""
    apply_theme()

    # Sidebar Navigation
    render_sidebar_brand()
    render_sidebar_route_navigation(active_page_id="support")

    service = get_service("support_chatbot")
    bundle, meta = load_active_bundle("support_chatbot")
    active_ver = meta.get("version", "v1") if isinstance(meta, dict) else "v1"
    render_sidebar_status_strip(active_ver=active_ver)

    # Page Header
    render_page_header(
        title="24/7 AI TRAVEL ASSISTANT & PLANNER",
        subtitle="Your all-in-one conversational railway co-pilot. Plan customized train trips, ask IRCTC refund & Tatkal policies in English & हिंदी, or track any 10-digit PNR live.",
    )

    # Sub-strip with live connection badges
    has_rapidapi = bool(os.getenv("RAPIDAPI_KEY") or os.getenv("X_RAPIDAPI_KEY"))
    rapidapi_label = "⚡ RapidAPI Live PNR: Active" if has_rapidapi else "⚡ RapidAPI: Verified Offline Fallback Ready"
    rapidapi_color = "#059669" if has_rapidapi else "#0B1F3A"

    st.markdown(
        f"""
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 1.25rem;">
            <span style="font-size: 0.76rem; background: rgba(11, 31, 58, 0.06); border: 1px solid rgba(11, 31, 58, 0.15); color: #0B1F3A; padding: 2px 10px; border-radius: 12px; font-weight: 600;">
                🌐 English &amp; हिंदी (देवनागरी) Supported
            </span>
            <span style="font-size: 0.76rem; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); color: {rapidapi_color}; padding: 2px 10px; border-radius: 12px; font-weight: 600;">
                {rapidapi_label}
            </span>
            <span style="font-size: 0.74rem; color: #5B6B82; margin-left: 4px;">
                Verified timetable database &amp; IRCTC passenger rule retrieval
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Route Builder Expander
    with st.expander("✨ Need Help Planning? Quick Route Builder Form (Click to open)", expanded=False):
        st.caption("Select your trip preferences below to generate an AI plan instantly:")
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            origin_sel = st.selectbox(
                "From (Origin)",
                ["Pune", "New Delhi", "Mumbai Central", "Bangalore", "Bhopal", "Varanasi", "Jaipur", "Lucknow", "Chandigarh"],
                index=0,
                key="rb_origin",
            )
        with b_c2:
            dest_sel = st.selectbox(
                "To (Destination)",
                ["Delhi", "Madgaon (Goa)", "Mumbai Central", "Chennai Central", "Bhopal", "Varanasi", "Jaipur", "Lucknow"],
                index=0,
                key="rb_dest",
            )
        with b_c3:
            budget_sel = st.select_slider(
                "Max Budget (₹)",
                options=[600, 900, 1200, 1500, 2000, 2500, 3000],
                value=1500,
                key="rb_budget",
            )
        with b_c4:
            deadline_sel = st.selectbox(
                "Arrive Before",
                ["Any time", "10:00 AM", "12:00 PM", "4:00 PM", "8:00 PM"],
                index=1,
                key="rb_deadline",
            )

        if st.button("🚀 Plan My Journey with AI", type="primary", use_container_width=True):
            deadline_str = f", reach before {deadline_sel}" if deadline_sel != "Any time" else ""
            st.session_state["pending_prompt"] = f"{origin_sel} to {dest_sel} under ₹{budget_sel}{deadline_str}"
            st.rerun()

    # Centered conversation container for maximum focus and readability
    col_left, col_center, col_right = st.columns([0.04, 0.92, 0.04])

    with col_center:
        # Initialize chat history in session state
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "Hello! I am your **24/7 Indian Railway Travel Assistant & Planner**. Here is what I can do for you:\n\n"
                        "- 🚆 **Smart Trip Planning**: Ask for any journey (e.g. *\"Pune to Delhi under ₹1500, reach before 10 AM\"* or *\"Mumbai to Goa under ₹900\"*)\n"
                        "- 🔍 **Live PNR Tracking**: Enter any 10-digit PNR (e.g. *\"Check PNR 9823471029\"*)\n"
                        "- 💰 **Refund & Cancellation**: Know exact deductions, clerkage, and refund timelines\n"
                        "- ⏰ **Tatkal Rules**: Timings for AC/Sleeper and dynamic surge policies\n"
                        "- 💺 **RAC & Waitlist**: Travel allowances, chart preparation times, and berth sharing\n\n"
                        "आप हिंदी (देवनागरी) में भी टिकट रद्दीकरण, रिफंड, यात्रा योजना या पीएनआर स्थिति से संबंधित प्रश्न पूछ सकते हैं।"
                    ),
                    "badge": "Verified Railway Assistant",
                    "source": "Platform System Manual",
                    "similarity": 1.0,
                    "is_grounded": True,
                    "is_escalated": False,
                    "masked_query": "System Welcome",
                    "feedback_given": None,
                }
            ]

        # Suggestion chips in clean, non-truncating white cards
        st.markdown(
            '<div style="font-size: 0.82rem; font-weight: 600; color: #5B6B82; margin-bottom: 0.4rem; margin-top: 0.5rem;">'
            'Quick questions & sample trip requests (click to test):'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🚆 Pune → Delhi under ₹1500, reach before 10 AM", key="chip_trip1", use_container_width=True):
                st.session_state["pending_prompt"] = "Pune to Delhi tomorrow under ₹1500, reach before 10 AM"
        with c2:
            if st.button("🚆 Mumbai → Goa Friday under ₹900", key="chip_trip2", use_container_width=True):
                st.session_state["pending_prompt"] = "Mumbai to Goa Friday night under ₹900"
        with c3:
            if st.button("🔍 Check Live PNR: 9823471029", key="chip_pnr", use_container_width=True):
                st.session_state["pending_prompt"] = "Please check status for PNR 9823471029"

        c4, c5, c6 = st.columns(3)
        with c4:
            if st.button("💰 Ticket cancellation refund rules", key="chip_refund", use_container_width=True):
                st.session_state["pending_prompt"] = "What is the refund rule on ticket cancellation?"
        with c5:
            if st.button("⏰ Tatkal booking timings & eligibility", key="chip_tatkal", use_container_width=True):
                st.session_state["pending_prompt"] = "What time does Tatkal booking start for AC and Sleeper?"
        with c6:
            if st.button("🇮🇳 हिंदी: कन्फर्म टिकट पर कितना रिफंड मिलता है?", key="chip_hindi", use_container_width=True):
                st.session_state["pending_prompt"] = "कन्फर्म टिकट कैंसिल करने पर कितना रिफंड मिलता है?"

        st.markdown('<div style="margin-top: 1.25rem;"></div>', unsafe_allow_html=True)

        # Render conversation history
        for idx, msg in enumerate(st.session_state["chat_messages"]):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(
                        '<div style="font-size: 0.74rem; font-weight: 700; color: #5B6B82; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem;">'
                        'YOU'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(
                        '<div style="font-size: 0.74rem; font-weight: 700; color: #0B1F3A; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 6px;">'
                        '<span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #10B981;"></span>'
                        'RAILWAY TRAVEL ASSISTANT'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    # 1. Render rich Trip card if trip_data present
                    if msg.get("trip_data"):
                        _render_trip_card(msg["trip_data"])

                    # 2. Render rich PNR card if pnr_data present
                    elif msg.get("pnr_data"):
                        _render_pnr_card(msg["pnr_data"])

                    # 3. Standard text response
                    else:
                        st.markdown(msg["content"])

                    # Reassurance tag and feedback
                    tag_color = "#059669" if msg.get("is_grounded", True) else "#D97706"
                    tag_bg = "rgba(5, 150, 105, 0.08)" if msg.get("is_grounded", True) else "rgba(217, 119, 6, 0.08)"
                    tag_badge = msg.get("badge") or ("Verified IRCTC Policy" if msg.get("is_grounded", True) else "Transferred to Railway Helpdesk")

                    f_col1, f_col2, f_col_spacer = st.columns([2.5, 1.5, 4.0])
                    with f_col1:
                        st.markdown(
                            f"""
                            <div style="display: inline-block; font-size: 0.74rem; font-weight: 600; color: {tag_color}; background: {tag_bg}; border: 1px solid {tag_color}40; border-radius: 12px; padding: 2px 10px; margin-top: 4px;">
                                ● {tag_badge}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Plain text feedback buttons
                    if idx > 0 and msg.get("feedback_given") is None:
                        with f_col2:
                            fb_c1, fb_c2 = st.columns(2)
                            with fb_c1:
                                if st.button("Helpful", key=f"helpful_{idx}", use_container_width=True):
                                    service.log_feedback(msg.get("masked_query", "Query"), is_helpful=True)
                                    msg["feedback_given"] = "Helpful"
                                    st.rerun()
                            with fb_c2:
                                if st.button("Not helpful", key=f"not_helpful_{idx}", use_container_width=True):
                                    service.log_feedback(msg.get("masked_query", "Query"), is_helpful=False)
                                    msg["feedback_given"] = "Not helpful"
                                    st.rerun()
                    elif msg.get("feedback_given"):
                        with f_col2:
                            st.caption(f"Feedback: {msg['feedback_given']}")

        # Input handling
        prompt = st.chat_input("Ask a question, track a 10-digit PNR, or say 'Pune to Delhi under ₹1500'...")
        if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
            prompt = st.session_state.pop("pending_prompt")

        if prompt:
            # 1. Append user message
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})

            # 2. Predict / retrieve response
            with st.spinner("Checking verified railway documentation & timetables..."):
                res = service.predict(prompt)

            # 3. Append assistant response
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": res["reply"],
                "badge": res["badge"],
                "source": res["source"],
                "similarity": res["similarity"],
                "is_grounded": res["is_grounded"],
                "is_escalated": res["is_escalated"],
                "masked_query": res["masked_query"],
                "trip_data": res.get("trip_data"),
                "pnr_data": res.get("pnr_data"),
                "feedback_given": None,
            })
            st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
