"""
dashboard/app.py
Home Page for YatraGov AI: "Departure Board & Ticket" Theme.
Contains the dark navy hero banner, ticket-style feature directory, and academic footer.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.config import APP_NAME, MODULE_CATALOG
from app.governance.audit import init_audit_db
from dashboard.components.theme import WARM_PAPER, apply_theme
from dashboard.components.ui import (
    load_active_bundle,
    render_footer,
    render_hero_banner,
    render_sidebar_brand,
    render_sidebar_route_navigation,
    render_sidebar_status_strip,
)

st.set_page_config(
    page_title=f"{APP_NAME} - Predictive Travel & Governance",
    page_icon="🚆",
    layout="wide",
)


def main():
    """Render the Home Page with Railway Departure Board and Ticket Theme."""
    apply_theme()
    init_audit_db()

    # Load active ticket model metadata for system runtime panel
    _, meta = load_active_bundle("ticket_predictor")
    active_ver = meta.get("version", "v1") if isinstance(meta, dict) else "v1"

    # Dark Navy Sidebar
    render_sidebar_brand()
    render_sidebar_route_navigation(active_page_id="home")
    render_sidebar_status_strip(active_ver=active_ver)

    # Hero Banner in Navy with Saffron Tagline & Track Line
    render_hero_banner()

    # Section Title: Active Journey Modules
    st.markdown(
        """
        <div class="rail-section-header" style="margin-bottom: 0.85rem;">
            JOURNEY MODULES & CONTROL SYSTEMS
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active Modules + Governance Console in 4 equal columns
    active_modules = [m for m in MODULE_CATALOG if m["built"]]
    active_cols = st.columns(len(active_modules) + 1)

    for col, mod in zip(active_cols[:len(active_modules)], active_modules):
        with col:
            st.markdown(
                f"""
                <div class="yg-ticket" style="min-height: 195px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 1.25rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: #0B1F3A;">
                            {mod['title']}
                        </div>
                        <div style="font-size: 0.84rem; color: #5B6B82; margin-top: 0.35rem; line-height: 1.35;">
                            {mod['desc']}
                        </div>
                    </div>
                    <div class="yg-ticket-tear" style="margin: 0.65rem 0;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.page_link(mod["page"], label=f"Open {mod['short_title']} →", use_container_width=True)

    with active_cols[-1]:
        st.markdown(
            """
            <div class="yg-ticket" style="min-height: 195px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 1.25rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: #0B1F3A;">
                        AI Governance Console
                    </div>
                    <div style="font-size: 0.84rem; color: #5B6B82; margin-top: 0.35rem; line-height: 1.35;">
                        Statistical covariate drift (PSI/KS), model catalog, rollback, and autonomous supervisor.
                    </div>
                </div>
                <div class="yg-ticket-tear" style="margin: 0.65rem 0;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/governance_console.py", label="Open Governance Console →", use_container_width=True)

    # Roadmap Modules (Greyed with 'Coming soon' stamp)
    st.markdown(
        """
        <div class="rail-section-header" style="margin-top: 2rem; margin-bottom: 0.85rem;">
            ROADMAP & EXPANSION PIPELINE
        </div>
        """,
        unsafe_allow_html=True,
    )

    roadmap_items = [m for m in MODULE_CATALOG if not m["built"]]
    road_cols = st.columns(len(roadmap_items))
    for col, mod in zip(road_cols, roadmap_items):
        with col:
            st.markdown(
                f"""
                <div class="yg-ticket" style="opacity: 0.85; background: {WARM_PAPER}; min-height: 165px; display: flex; flex-direction: column; justify-content: space-between; border-style: dashed;">
                    <div>
                        <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 1.1rem; font-weight: 700; text-transform: uppercase; color: #0B1F3A; line-height: 1.2;">
                            {mod['title']}
                        </div>
                        <div style="font-size: 0.82rem; color: #5B6B82; margin-top: 0.35rem; line-height: 1.35;">
                            {mod['desc']}
                        </div>
                    </div>
                    <div>
                        <span class="rail-badge-soon" style="display: inline-block; margin-top: 0.5rem;">Coming soon</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_footer()


if __name__ == "__main__":
    main()
