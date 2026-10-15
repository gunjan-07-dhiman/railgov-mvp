"""
dashboard/app.py
Home page and feature overview for the RailGov Streamlit application.
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
from dashboard.components.theme import render_header, render_sidebar_system_info

st.set_page_config(
    page_title="RailGov - Home",
    page_icon="🚆",
    layout="wide",
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
    """Render the Home Page with project overview and feature directory."""
    init_audit_db()
    _, meta = load_ticket_bundle()
    active_ver = meta.get("version", "v1") if isinstance(meta, dict) else "v1"

    render_header(
        title="RailGov - Indian Railways AI Predictor & Governance System",
        subtitle="Autonomous AI governance and passenger prediction layer for Indian Railways.",
        icon="🚆",
    )
    render_sidebar_system_info(active_ver)

    st.markdown("### 🗺️ Project Capabilities & Modules")
    st.markdown("Use the navigation menu in the left sidebar to access active modules:")

    features = [
        {"Icon": "🎫", "Module": "Ticket Confirmation Predictor", "Status": "Active (Built)", "Description": "Predict waitlist and RAC confirmation probabilities with TreeSHAP log-odds explainability."},
        {"Icon": "🛡️", "Module": "AI Governance Console", "Status": "Active (Built)", "Description": "Central command center for statistical drift (PSI/KS), model catalog, rollback, append-only audit, and autonomous supervisor."},
        {"Icon": "⏱️", "Module": "Delay & Congestion Forecasting", "Status": "Coming soon (Step 2)", "Description": "Delay minutes prediction, station platform congestion levels, and proactive passenger delay alerts."},
        {"Icon": "💰", "Module": "Dynamic Fare Movement", "Status": "Coming soon (Step 3)", "Description": "Demand surge estimation, best booking window recommendations, and price anomaly detection."},
        {"Icon": "🔍", "Module": "Booking Fraud & Bot Protection", "Status": "Coming soon (Step 4)", "Description": "High-frequency suspicious booking risk scores with explainable risk flags."},
        {"Icon": "💬", "Module": "Multilingual Support Chatbot", "Status": "Coming soon (Step 5)", "Description": "Grounded FAQ retrieval with Indian language script detection and grounding guardrails."},
        {"Icon": "🧭", "Module": "Alternative Travel Recommender", "Status": "Coming soon (Step 6)", "Description": "Ranking alternative train routes with explainability and algorithmic fairness audits."},
        {"Icon": "🤖", "Module": "Autonomous Journey Planner", "Status": "Coming soon (Step 7)", "Description": "Rule-based autonomous orchestrator coordinating cross-module bookings with user approval gates."},
    ]

    import pandas as pd
    st.dataframe(pd.DataFrame(features), use_container_width=True, hide_index=True)

    st.info("👈 Select a feature page from the sidebar menu to get started.")


main()
