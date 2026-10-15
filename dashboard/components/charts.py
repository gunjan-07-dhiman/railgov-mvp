"""
dashboard/components/charts.py
Reusable Plotly chart components for the RailGov dashboard.
"""

from typing import Any, Dict
import plotly.graph_objects as go
from app.config import PSI_CRITICAL_THRESHOLD, PSI_WARNING_THRESHOLD


def create_psi_bar_chart(drift_report: Dict[str, Any]) -> go.Figure:
    """Generate a Plotly bar chart of feature PSI values with severity color coding."""
    feature_metrics = drift_report.get("feature_metrics", {})
    if not feature_metrics:
        fig = go.Figure()
        fig.update_layout(title="No drift metrics available.")
        return fig

    features = list(feature_metrics.keys())
    psi_values = [feature_metrics[f]["psi"] for f in features]

    colors = []
    for psi in psi_values:
        if psi >= PSI_CRITICAL_THRESHOLD:
            colors.append("#ef4444")  # Red: Critical
        elif psi >= PSI_WARNING_THRESHOLD:
            colors.append("#f97316")  # Orange: Warning
        else:
            colors.append("#10b981")  # Green: Normal

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=features,
            y=psi_values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in psi_values],
            textposition="auto",
        )
    )

    # Reference lines for Warning and Critical PSI thresholds
    fig.add_hline(
        y=PSI_WARNING_THRESHOLD,
        line_dash="dash",
        line_color="#f97316",
        annotation_text="Warning (0.10)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=PSI_CRITICAL_THRESHOLD,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Critical (0.20)",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Population Stability Index (PSI) per Feature",
        xaxis_title="Feature",
        yaxis_title="PSI Value",
        yaxis=dict(range=[0, max(max(psi_values) * 1.25, 0.35)]),
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
