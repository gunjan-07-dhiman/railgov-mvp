"""
dashboard/components/charts.py
Reusable Plotly chart components for YatraGov AI:
- Feature PSI severity bar chart.
- Half-gauge probability indicator for ticket confirmation.
"""

from typing import Any, Dict
import plotly.graph_objects as go
from app.config import PSI_CRITICAL_THRESHOLD, PSI_WARNING_THRESHOLD
from dashboard.components.theme import (
    FONT_HEADINGS,
    FONT_MONO,
    NAVY,
    SAFFRON,
    SIGNAL_AMBER,
    SIGNAL_GREEN,
    SIGNAL_RED,
    WARM_PAPER,
)


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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NAVY, family="Barlow, sans-serif"),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_confirmation_gauge(prob_pct: float, status_label: str) -> go.Figure:
    """Render a half-gauge meter calibrated in railway signal colors."""
    if status_label == "Likely Confirmed":
        gauge_color = SIGNAL_GREEN
    elif status_label == "Uncertain":
        gauge_color = SIGNAL_AMBER
    else:
        gauge_color = SIGNAL_RED

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob_pct, 1),
            number={"suffix": "%", "font": {"family": "Share Tech Mono", "size": 38, "color": NAVY}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": NAVY, "nticks": 5},
                "bar": {"color": gauge_color, "thickness": 0.35},
                "bgcolor": "#E2E8F0",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.12)"},
                    {"range": [40, 70], "color": "rgba(250, 204, 21, 0.12)"},
                    {"range": [70, 100], "color": "rgba(34, 197, 94, 0.12)"},
                ],
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig


def create_fraud_risk_gauge(risk_score: float, level: str) -> go.Figure:
    """Render a risk gauge (0-100) calibrated in railway signal colors."""
    if level == "Low":
        bar_color = SIGNAL_GREEN
    elif level == "Medium":
        bar_color = SIGNAL_AMBER
    else:
        bar_color = SIGNAL_RED

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(risk_score, 0),
            number={"suffix": " / 100", "font": {"family": "Share Tech Mono", "size": 34, "color": NAVY}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": NAVY, "nticks": 5},
                "bar": {"color": bar_color, "thickness": 0.35},
                "bgcolor": "#E2E8F0",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(34, 197, 94, 0.12)"},
                    {"range": [30, 70], "color": "rgba(250, 204, 21, 0.12)"},
                    {"range": [70, 100], "color": "rgba(239, 68, 68, 0.12)"},
                ],
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig


def create_fare_horizon_chart(
    days_series: list,
    fares_series: list,
    current_days: int,
    best_day: int,
    best_fare: int,
    today_fare: int,
) -> go.Figure:
    """Render fare trajectory across booking days (navy line, saffron best day marker)."""
    fig = go.Figure()

    # Trajectory curve
    fig.add_trace(
        go.Scatter(
            x=days_series,
            y=fares_series,
            mode="lines",
            line=dict(color=NAVY, width=2.5),
            name="Estimated Fare",
            hovertemplate="Day %{x} ahead: ₹%{y:,}<extra></extra>",
        )
    )

    # Current Day Marker
    fig.add_trace(
        go.Scatter(
            x=[current_days],
            y=[today_fare],
            mode="markers+text",
            marker=dict(color=NAVY, size=10, symbol="circle", line=dict(color="#FFFFFF", width=2)),
            text=[f"Today (₹{today_fare:,})"],
            textposition="top center",
            name="Today",
            hoverinfo="skip",
        )
    )

    # Best Booking Day Marker (Saffron)
    fig.add_trace(
        go.Scatter(
            x=[best_day],
            y=[best_fare],
            mode="markers+text",
            marker=dict(color=SAFFRON, size=13, symbol="diamond", line=dict(color=NAVY, width=2)),
            text=[f"Best Day {best_day} (₹{best_fare:,})"],
            textposition="bottom center",
            name="Best Day",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Fare Trajectory Across Booking Horizon (Days Ahead)",
        xaxis=dict(
            title="Days Before Departure (Horizon: 0–60)",
            tickmode="linear",
            dtick=10,
            gridcolor="rgba(143, 163, 191, 0.2)",
            autorange="reversed",  # 60 days on left -> 0 days (departure) on right
        ),
        yaxis=dict(
            title="Estimated Fare (INR)",
            gridcolor="rgba(143, 163, 191, 0.2)",
            tickprefix="₹",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NAVY, family="Barlow, sans-serif"),
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
    )
    return fig
