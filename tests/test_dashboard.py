"""
tests/test_dashboard.py
Tests for Streamlit governance dashboard helper functions:
- Plotly PSI bar chart rendering and severity color mapping.
- Model bundle loader and registry integration.
- Clean module import without syntax or runtime initialization errors.
"""

import plotly.graph_objects as go
import pytest

from dashboard.app import create_psi_bar_chart, load_active_bundle


def test_dashboard_import_cleanly():
    import dashboard.app as dash_app
    assert hasattr(dash_app, "main")
    assert hasattr(dash_app, "create_psi_bar_chart")


def test_create_psi_bar_chart_valid():
    sample_drift_report = {
        "overall_severity": "critical",
        "feature_metrics": {
            "wl_position": {"psi": 0.28, "severity": "critical", "feature_type": "numeric"},
            "is_festival_season": {"psi": 0.35, "severity": "critical", "feature_type": "numeric"},
            "route_popularity": {"psi": 0.14, "severity": "warning", "feature_type": "numeric"},
            "day_of_week": {"psi": 0.02, "severity": "none", "feature_type": "numeric"},
        },
    }

    fig = create_psi_bar_chart(sample_drift_report)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == 4
    # Check that critical bars receive red (#ef4444)
    colors = fig.data[0].marker.color
    assert "#ef4444" in colors
    assert "#f97316" in colors
    assert "#10b981" in colors


def test_create_psi_bar_chart_empty():
    fig = create_psi_bar_chart({})
    assert isinstance(fig, go.Figure)


def test_load_active_bundle():
    bundle, meta = load_active_bundle()
    if bundle is not None:
        assert "model" in bundle
        assert "preprocessor" in bundle
