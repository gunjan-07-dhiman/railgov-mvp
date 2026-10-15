"""
dashboard/app.py
Entry point and navigation router for the RailGov Streamlit application.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.base import get_service, list_services
import app.modules.ticket_predictor  # Ensure ticket predictor service registers
from app.governance.audit import init_audit_db
from app.governance.registry import get_active, register
from dashboard.components.charts import create_psi_bar_chart
from dashboard.pages.governance_console import render_governance_console_page
from dashboard.pages.ticket_confirmation import render_ticket_confirmation_page

st.set_page_config(
    page_title="RailGov - Indian Railways AI Predictor & Governance",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_ticket_bundle():
    """Load or bootstrap active model bundle for the ticket predictor."""
    service = get_service("ticket_predictor")
    active = get_active("ticket_predictor")
    if not active:
        try:
            base_df = service.load_baseline_df()
            res = service.train(base_df, version="v1")
            register(
                version="v1",
                metrics=res["metrics"],
                training_data_desc="Bootstrap baseline model v1",
                status="active",
                model_name="ticket_predictor",
            )
            active = get_active("ticket_predictor")
        except Exception:
            return None, {}

    try:
        bundle, meta = service.load_model(active["version"])
        return bundle, meta
    except Exception:
        return None, {}


load_active_bundle = load_ticket_bundle


def main():
    """Application entry point and sidebar mode router."""
    init_audit_db()
    bundle, meta = load_ticket_bundle()
    active_ver = meta.get("version", "v1") if isinstance(meta, dict) else "v1"

    st.sidebar.title("🚆 RailGov System")
    portal_mode = st.sidebar.radio(
        "Select Portal Mode",
        [
            "🎫 Passenger Portal",
            "🛡️ AI Governance Console",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 System Info")
    st.sidebar.markdown(f"- **Active Model**: `{active_ver}`")
    st.sidebar.markdown(f"- **Registered Services**: {len(list_services())}")
    st.sidebar.markdown("- **Explainability**: TreeSHAP (Log-odds)")
    st.sidebar.markdown("- **Audit Log**: SQLite (Append-Only)")

    if "Passenger Portal" in portal_mode:
        render_ticket_confirmation_page(bundle, active_ver)
    else:
        render_governance_console_page()


if __name__ == "__main__":
    main()
