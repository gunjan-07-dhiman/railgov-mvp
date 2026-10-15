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

    # Section Title: The 6 Core Travel Features
    st.markdown(
        """
        <div class="rail-section-header" style="margin-bottom: 0.85rem;">
            THE 6 CORE AI TRAVEL FEATURES
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean 3-column responsive grid for the 6 features
    cols_per_row = 3
    for i in range(0, len(MODULE_CATALOG), cols_per_row):
        row_mods = MODULE_CATALOG[i : i + cols_per_row]
        row_cols = st.columns(cols_per_row)
        for col, mod in zip(row_cols, row_mods):
            with col:
                num = mod.get("num", i + 1)
                is_built = mod.get("built", False)
                opacity_style = "opacity: 1;" if is_built else "opacity: 0.85; border-style: dashed;"
                badge_html = f'<span style="font-family: \'Share Tech Mono\', monospace; font-size: 0.72rem; color: #10B981; font-weight: 700;">FEATURE #{num} • ACTIVE</span>' if is_built else f'<span class="rail-badge-soon">FEATURE #{num} • SOON</span>'

                st.markdown(
                    f"""
                    <div class="yg-ticket" style="{opacity_style} min-height: 185px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                                {badge_html}
                            </div>
                            <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 1.25rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: #0B1F3A;">
                                {mod['title']}
                            </div>
                            <div style="font-size: 0.85rem; color: #5B6B82; margin-top: 0.35rem; line-height: 1.35;">
                                {mod['desc']}
                            </div>
                        </div>
                        <div class="yg-ticket-tear" style="margin: 0.65rem 0;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if is_built:
                    st.page_link(mod["page"], label=f"Open {mod['short_title']} →", use_container_width=True)
                else:
                    st.caption("Module pipeline planned in expansion cycle.")

    # Governance Console
    st.markdown(
        """
        <div class="rail-section-header" style="margin-top: 2rem; margin-bottom: 0.85rem;">
            AI GOVERNANCE & AUDIT CONTROLS
        </div>
        """,
        unsafe_allow_html=True,
    )
    gov_col1, gov_col2 = st.columns([1.5, 1.5])
    with gov_col1:
        st.markdown(
            f"""
            <div class="yg-ticket" style="min-height: 160px;">
                <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 1.25rem; font-weight: 700; text-transform: uppercase; color: #0B1F3A;">
                    Governance Console
                </div>
                <div style="font-size: 0.85rem; color: #5B6B82; margin-top: 0.35rem; line-height: 1.35;">
                    Central oversight for statistical covariate drift (PSI/KS), model catalog, instant rollback, and autonomous health supervisor.
                </div>
                <div class="yg-ticket-tear" style="margin: 0.65rem 0;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/governance_console.py", label="Open Governance Console →", use_container_width=True)

    render_footer()


if __name__ == "__main__":
    main()
