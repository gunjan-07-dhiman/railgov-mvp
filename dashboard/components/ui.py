"""
dashboard/components/ui.py
Reusable UI components for YatraGov AI: "Departure Board & Ticket" Theme.
- Sidebar brand, route-line navigation, and control-room signal status strip.
- Departure board, ticket stub (with half-gauge), signal lights, hero banner, and headers.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

from app.base import get_service, list_services
from app.config import (
    APP_NAME,
    FOOTER_DISCLAIMER,
    HOME_TAGLINE,
    MODULE_CATALOG,
    SIDEBAR_TAGLINE,
)
from app.governance.audit import get_events, init_audit_db
from app.governance.registry import get_active, register
from dashboard.components.charts import create_confirmation_gauge
from dashboard.components.theme import (
    FONT_HEADINGS,
    FONT_MONO,
    MUTED_DARK,
    MUTED_LIGHT,
    NAVY,
    NAVY_BORDER,
    NAVY_SURFACE,
    SAFFRON,
    SIGNAL_AMBER,
    SIGNAL_GREEN,
    SIGNAL_RED,
    WHITE_CARD,
)


def load_active_bundle(model_name: str = "ticket_predictor") -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Load or bootstrap active model bundle for a governed service."""
    service = get_service(model_name)
    active = get_active(model_name)
    if not active:
        try:
            base_df = service.load_baseline_df()
            res = service.train(base_df, version="v1")
            register(
                version="v1",
                metrics=res["metrics"],
                training_data_desc="Bootstrap baseline model v1",
                status="active",
                model_name=model_name,
            )
            active = get_active(model_name)
        except Exception:
            return None, {}

    try:
        bundle, meta = service.load_model(active["version"])
        return bundle, meta
    except Exception:
        return None, {}


def render_sidebar_brand() -> None:
    """Render the railway signage brand block at the top of the sidebar."""
    st.sidebar.markdown(
        f"""
        <div style="padding: 0.25rem 0.5rem 0.75rem 0.5rem; margin-bottom: 0.75rem;">
            <div style="font-family: {FONT_HEADINGS}; font-size: 1.65rem; font-weight: 800; color: #FFFFFF; letter-spacing: 0.06em; text-transform: uppercase; line-height: 1.1;">
                {APP_NAME}
            </div>
            <div style="width: 38px; height: 3px; background-color: {SAFFRON}; margin-top: 5px; margin-bottom: 6px;"></div>
            <div style="font-family: {FONT_HEADINGS}; font-size: 0.72rem; font-weight: 700; color: {MUTED_DARK}; letter-spacing: 0.12em; text-transform: uppercase;">
                {SIDEBAR_TAGLINE}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_route_navigation(active_page_id: str) -> None:
    """
    Render clean route-line station navigation down the dark navy sidebar.
    Each row has an active indicator, station dot, clean labels, and non-clickable SOON badges.
    """
    # 1. MAIN
    st.sidebar.markdown('<div class="rail-sidebar-section">MAIN</div>', unsafe_allow_html=True)
    is_home = active_page_id == "home"
    st.sidebar.page_link(
        "app.py",
        label="●  Home & Overview" if is_home else "○  Home & Overview",
        use_container_width=True,
    )

    # 2. PASSENGER SERVICES
    st.sidebar.markdown('<div class="rail-sidebar-section">PASSENGER SERVICES</div>', unsafe_allow_html=True)
    for mod in MODULE_CATALOG:
        if mod["built"]:
            is_active = active_page_id == mod["id"]
            st.sidebar.page_link(
                mod["page"],
                label=f"{'●' if is_active else '○'}  {mod['short_title']}",
                use_container_width=True,
            )
        else:
            st.sidebar.markdown(
                f"""
                <div class="rail-nav-disabled">
                    <div style="display: flex; align-items: center;">
                        <span class="rail-dot rail-dot-soon"></span>
                        <span>{mod['short_title']}</span>
                    </div>
                    <span class="rail-badge-soon">SOON</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 3. AI GOVERNANCE
    st.sidebar.markdown('<div class="rail-sidebar-section">AI GOVERNANCE</div>', unsafe_allow_html=True)
    is_gov = active_page_id == "governance"
    st.sidebar.page_link(
        "pages/governance_console.py",
        label="●  Governance Console" if is_gov else "○  Governance Console",
        use_container_width=True,
    )


def render_sidebar_status_strip(active_ver: Optional[str] = "v1") -> None:
    """Render compact control-room signal status strip with real values only."""
    init_audit_db()
    services = list_services()
    model_count = len(services)

    try:
        drift_events = get_events(event_type="DRIFT_DETECTED", limit=50)
        unresolved_alerts = len([e for e in drift_events if "critical" in str(e.get("details", "")).lower()])
    except Exception:
        unresolved_alerts = 0

    alert_lamp = SIGNAL_GREEN if unresolved_alerts == 0 else SIGNAL_RED
    alert_text = "NORMAL" if unresolved_alerts == 0 else f"{unresolved_alerts} ALERTS"

    st.sidebar.markdown(
        f"""
        <div style="margin-top: 1.5rem; padding: 0.85rem; border-radius: 8px; background-color: {NAVY_SURFACE}; border: 1px solid {NAVY_BORDER}; font-size: 0.8rem;">
            <div style="font-family: {FONT_HEADINGS}; font-size: 0.72rem; font-weight: 700; color: {MUTED_DARK}; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.45rem;">
                SIGNAL STATUS & RUNTIME
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                <span style="color: {MUTED_DARK};">Governed Models:</span>
                <span style="color: #FFFFFF; font-weight: 700; font-family: {FONT_MONO};">{model_count} ACTIVE</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                <span style="color: {MUTED_DARK};">Track Drift:</span>
                <span style="display: inline-flex; align-items: center; color: #FFFFFF; font-weight: 600; font-size: 0.76rem;">
                    <span class="yg-signal-lamp" style="background-color: {alert_lamp};"></span>
                    {alert_text}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 0.35rem; border-top: 1px solid rgba(143, 163, 191, 0.15);">
                <span style="color: {MUTED_DARK};">Audit Log:</span>
                <span style="color: #FFFFFF; font-family: {FONT_MONO}; font-size: 0.75rem;">SQLITE WAL</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def signal_light(status: str, label: Optional[str] = None) -> str:
    """Return an HTML snippet for a real railway signal lamp with label."""
    s = status.lower()
    if any(k in s for k in ["normal", "green", "healthy", "likely confirmed", "on-time", "promoted"]):
        color = SIGNAL_GREEN
        tag = label or "NORMAL"
    elif any(k in s for k in ["warning", "amber", "moderate", "uncertain", "caution"]):
        color = SIGNAL_AMBER
        tag = label or "CAUTION"
    else:
        color = SIGNAL_RED
        tag = label or "CRITICAL"

    return f'<span style="display: inline-flex; align-items: center; font-weight: 600; font-size: 0.82rem;"><span class="yg-signal-lamp" style="background-color: {color};"></span>{tag}</span>'


def departure_board(rows: List[Tuple[str, str, Optional[str]]]) -> None:
    """
    Render railway departure board: dark navy panel, amber Share Tech Mono text.
    rows: list of (label, value, optional_status)
    """
    rows_html = ""
    for label, val, status in rows:
        val_color = SAFFRON
        if status == "green":
            val_color = SIGNAL_GREEN
        elif status == "red":
            val_color = SIGNAL_RED

        rows_html += f"""
        <div class="yg-board-row">
            <span class="yg-board-label">{label}</span>
            <span class="yg-board-val" style="color: {val_color};">{val}</span>
        </div>
        """

    st.markdown(
        f"""
        <div class="yg-departure-board">
            <div class="yg-board-header">DEPARTURE TELEMETRY & ARRIVAL FORECAST</div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def ticket_stub(status_str: str, prob_pct: float, details: Dict[str, Any], top_factors: List[Dict[str, Any]]) -> None:
    """
    Render ticket-shaped result card with notched sides and dashed tear line.
    Includes a Plotly half-gauge indicator.
    """
    if status_str == "Likely Confirmed":
        badge_bg = SIGNAL_GREEN
        badge_text_color = "#FFFFFF"
        header_text = "CONFIRMED STATUS: HIGH PROBABILITY"
    elif status_str == "Uncertain":
        badge_bg = SIGNAL_AMBER
        badge_text_color = NAVY
        header_text = "CONFIRMED STATUS: MODERATE / UNCERTAIN"
    else:
        badge_bg = SIGNAL_RED
        badge_text_color = "#FFFFFF"
        header_text = "CONFIRMED STATUS: LOW PROBABILITY"

    # Ticket top half
    st.markdown(
        f"""
        <div class="yg-ticket">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <span style="font-family: {FONT_HEADINGS}; font-size: 0.95rem; font-weight: 700; color: {NAVY}; letter-spacing: 0.05em; text-transform: uppercase;">
                    PASSENGER BOARDING ADVICE
                </span>
                <span style="background-color: {badge_bg}; color: {badge_text_color}; font-family: {FONT_HEADINGS}; font-size: 0.78rem; font-weight: 700; padding: 4px 10px; border-radius: 4px; letter-spacing: 0.06em; text-transform: uppercase;">
                    {status_str.upper()}
                </span>
            </div>
            <div style="font-size: 0.86rem; color: {MUTED_LIGHT}; margin-bottom: 0.5rem;">
                {details.get("desc", "")}
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Half-gauge meter
    gauge_fig = create_confirmation_gauge(prob_pct, status_str)
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Dashed tear line & bottom stub
    factors_html = "".join([f"<li style='margin-bottom: 0.35rem;'><strong>{f['explanation']}</strong></li>" for f in top_factors])
    st.markdown(
        f"""
            <div class="yg-ticket-tear"></div>
            <div style="font-family: {FONT_HEADINGS}; font-size: 0.88rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: {NAVY}; margin-bottom: 0.4rem;">
                DECIDING FACTORS (TREESHAP EXPLAINABILITY)
            </div>
            <ul style="padding-left: 1.25rem; font-size: 0.85rem; color: {NAVY}; margin-bottom: 0;">
                {factors_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str) -> None:
    """Render uppercase title in Barlow Condensed with short saffron underline. No emojis."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div class="rail-page-title">{title}</div>
            <div style="font-size: 0.88rem; color: {MUTED_LIGHT}; margin-top: 0.15rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_banner() -> None:
    """Render dark navy railway hero banner with greeting, saffron tagline, and dashed track line."""
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        greeting_text = "GOOD MORNING"
    elif 12 <= current_hour < 17:
        greeting_text = "GOOD AFTERNOON"
    else:
        greeting_text = "GOOD EVENING"

    st.markdown(
        f"""
        <div style="background-color: {NAVY}; border-radius: 12px; padding: 2rem 2.25rem; color: #FFFFFF; margin-bottom: 2rem; border-bottom: 4px dashed {SAFFRON};">
            <div style="font-family: {FONT_HEADINGS}; font-size: 2.1rem; font-weight: 800; letter-spacing: 0.05em; line-height: 1.1;">
                {greeting_text}, TRAVELER
            </div>
            <div style="font-family: {FONT_HEADINGS}; font-size: 0.95rem; font-weight: 700; color: {SAFFRON}; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.45rem;">
                {HOME_TAGLINE}
            </div>
            <div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 0.5rem; max-width: 680px;">
                Autonomous multi-model AI governance and predictive journey intelligence for Indian Railways passenger travel contexts.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render academic prototype disclaimer footer."""
    st.markdown(
        f"""
        <div class="yg-footer">
            {FOOTER_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )
