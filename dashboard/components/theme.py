"""
dashboard/components/theme.py
Design tokens, color palettes, and styling utilities for Streamlit dashboard.
"""

import streamlit as st

# Color Palette
PRIMARY_COLOR = "#1e3a8a"      # Deep Navy
SECONDARY_COLOR = "#0284c7"    # Sky Blue
SUCCESS_COLOR = "#10b981"      # Emerald Green
WARNING_COLOR = "#f59e0b"      # Amber
DANGER_COLOR = "#ef4444"       # Rose Red
BG_CARD_LIGHT = "#f8fafc"      # Slate Light


def render_header(title: str, subtitle: str, icon: str = "🚆") -> None:
    """Render a consistent page header with title, subtitle, and icon."""
    st.markdown(
        "<style>[data-testid='stSidebarNav'] {display: none;}</style>",
        unsafe_allow_html=True,
    )
    st.markdown(f"## {icon} {title}")
    st.caption(subtitle)
    st.markdown("---")


def status_badge(status: str) -> str:
    """Return an emoji and color tag for a high-level status string."""
    status_lower = status.lower()
    if any(w in status_lower for w in ["healthy", "normal", "active", "likely confirmed", "promoted"]):
        return f"🟢 **{status.upper()}**"
    if any(w in status_lower for w in ["warning", "uncertain", "moderate", "standby"]):
        return f"🟡 **{status.upper()}**"
    return f"🔴 **{status.upper()}**"
