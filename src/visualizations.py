"""
visualizations.py — All Plotly chart builders for the EV Demand Forecaster.
"""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from src.config import COLORS, CHART_COLORS, PLOTLY_LAYOUT


def _base_fig(**kwargs) -> go.Figure:
    """Create a figure pre-configured with our theme."""
    fig = go.Figure()
    layout = {**PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig


# ─── Forecast Chart with Confidence Interval ─────────────────────────────────

def plot_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    county: str,
    horizon_label: str = "3 Years",
) -> go.Figure:
    """Combined historical + forecast line chart with confidence band."""
    fig = _base_fig(
        title=dict(text=f"Cumulative EV Adoption — {county} County", font=dict(size=18)),
        height=480,
    )

    # Historical line
    fig.add_trace(go.Scatter(
        x=historical_df["Date"],
        y=historical_df["Cumulative_EV"],
        mode="lines+markers",
        name="Historical",
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=4),
        hovertemplate="<b>%{x|%b %Y}</b><br>Cumulative EVs: %{y:,.0f}<extra></extra>",
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(forecast_df["Date"]) + list(forecast_df["Date"][::-1]),
        y=list(forecast_df["Cumulative_Upper"]) + list(forecast_df["Cumulative_Lower"][::-1]),
        fill="toself",
        fillcolor="rgba(16,185,129,0.12)",
        line=dict(width=0),
        name="80% Confidence",
        hoverinfo="skip",
        showlegend=True,
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Cumulative_EV"],
        mode="lines+markers",
        name=f"Forecast ({horizon_label})",
        line=dict(color=COLORS["accent_blue"], width=2.5, dash="dot"),
        marker=dict(size=4),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>",
    ))

    # Divider line at transition point
    last_hist_date = historical_df["Date"].iloc[-1]
    fig.add_shape(
        type="line",
        x0=last_hist_date, x1=last_hist_date,
        y0=0, y1=1, yref="paper",
        line=dict(color=COLORS["text_muted"], width=1, dash="dash"),
    )
    fig.add_annotation(
        x=last_hist_date, y=1.05, yref="paper",
        text="Forecast Start",
        showarrow=False,
        font=dict(color=COLORS["text_secondary"], size=11),
    )

    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Cumulative EV Count")
    return fig


# ─── Monthly Forecast Bar Chart ─────────────────────────────────────────────

def plot_monthly_forecast(forecast_df: pd.DataFrame, county: str) -> go.Figure:
    """Bar chart of monthly predicted EV registrations with CI error bars."""
    fig = _base_fig(
        title=dict(text=f"Monthly EV Forecast — {county} County", font=dict(size=16)),
        height=380,
    )

    fig.add_trace(go.Bar(
        x=forecast_df["Date"],
        y=forecast_df["Predicted_EV_Total"],
        marker=dict(
            color=forecast_df["Predicted_EV_Total"],
            colorscale=[[0, COLORS["accent_dark"]], [1, COLORS["accent_light"]]],
            line=dict(width=0),
            cornerradius=4,
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=(forecast_df["Upper_CI"] - forecast_df["Predicted_EV_Total"]).tolist(),
            arrayminus=(forecast_df["Predicted_EV_Total"] - forecast_df["Lower_CI"]).tolist(),
            color=COLORS["text_muted"],
            thickness=1,
        ),
        name="Monthly Forecast",
        hovertemplate="<b>%{x|%b %Y}</b><br>Predicted: %{y:,.0f}<extra></extra>",
    ))

    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="EV Registrations")
    return fig


# ─── BEV vs PHEV Donut ──────────────────────────────────────────────────────

def plot_bev_phev_split(county_df: pd.DataFrame, county: str) -> go.Figure:
    """Donut chart of BEV vs PHEV split."""
    total_bev = county_df["Battery Electric Vehicles (BEVs)"].sum()
    total_phev = county_df["Plug-In Hybrid Electric Vehicles (PHEVs)"].sum()

    fig = go.Figure(go.Pie(
        labels=["Battery EV (BEV)", "Plug-in Hybrid (PHEV)"],
        values=[total_bev, total_phev],
        hole=0.6,
        marker=dict(colors=[COLORS["accent"], COLORS["accent_purple"]],
                    line=dict(color=COLORS["bg_primary"], width=2)),
        textinfo="percent+label",
        textfont=dict(color=COLORS["text_primary"], size=13),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))

    layout = {**PLOTLY_LAYOUT}
    layout.update(
        title=dict(text=f"EV Type Breakdown — {county}", font=dict(size=16)),
        height=360,
        showlegend=False,
        annotations=[dict(
            text=f"<b>{total_bev + total_phev:,.0f}</b><br>Total",
            x=0.5, y=0.5, font_size=16,
            font_color=COLORS["text_primary"],
            showarrow=False,
        )],
    )
    fig.update_layout(**layout)
    return fig


# ─── Multi-County Comparison ─────────────────────────────────────────────────

def plot_comparison(comparison_data: list[dict]) -> go.Figure:
    """Overlay cumulative EV trends for multiple counties."""
    fig = _base_fig(
        title=dict(text="Cumulative EV Adoption — County Comparison", font=dict(size=18)),
        height=500,
    )

    for i, item in enumerate(comparison_data):
        color = CHART_COLORS[i % len(CHART_COLORS)]
        county = item["county"]

        # Historical
        fig.add_trace(go.Scatter(
            x=item["hist_dates"], y=item["hist_cum"],
            mode="lines", name=f"{county} (Historical)",
            line=dict(color=color, width=2.5),
            legendgroup=county,
            hovertemplate=f"<b>{county}</b><br>" + "%{x|%b %Y}<br>Cumulative: %{y:,.0f}<extra></extra>",
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=item["fc_dates"], y=item["fc_cum"],
            mode="lines", name=f"{county} (Forecast)",
            line=dict(color=color, width=2.5, dash="dot"),
            legendgroup=county,
            hovertemplate=f"<b>{county}</b><br>" + "%{x|%b %Y}<br>Forecast: %{y:,.0f}<extra></extra>",
        ))

    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Cumulative EV Count")
    return fig


# ─── State Choropleth Map ────────────────────────────────────────────────────

def plot_state_map(state_agg: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    """US state choropleth map."""
    latest = state_agg.loc[state_agg.groupby("State")["Date"].idxmax()]

    fig = go.Figure(go.Choropleth(
        locations=latest["State"],
        z=latest[value_col],
        locationmode="USA-states",
        colorscale=[
            [0.0, COLORS["bg_elevated"]],
            [0.3, COLORS["accent_dark"]],
            [0.6, COLORS["accent"]],
            [1.0, COLORS["accent_light"]],
        ],
        colorbar=dict(
            title=dict(text=value_col.replace("_", " "), font=dict(color=COLORS["text_secondary"])),
            tickfont=dict(color=COLORS["text_secondary"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        marker_line_color=COLORS["border"],
        marker_line_width=0.5,
        hovertemplate="<b>%{location}</b><br>" + f"{value_col}: " + "%{z:,.0f}<extra></extra>",
    ))

    layout = {**PLOTLY_LAYOUT}
    layout.update(
        title=dict(text=title, font=dict(size=18)),
        geo=dict(
            scope="usa",
            bgcolor=COLORS["bg_primary"],
            lakecolor=COLORS["bg_primary"],
            landcolor=COLORS["bg_secondary"],
            showlakes=True,
            showframe=False,
        ),
        height=450,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.update_layout(**layout)
    return fig


# ─── Monthly Heatmap ─────────────────────────────────────────────────────────

def plot_monthly_heatmap(county_df: pd.DataFrame, county: str) -> go.Figure:
    """Year × Month heatmap of EV registrations."""
    pivot = county_df.pivot_table(
        index="year", columns="month",
        values="Electric Vehicle (EV) Total",
        aggfunc="sum", fill_value=0,
    )
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[month_labels[c - 1] for c in pivot.columns],
        y=pivot.index.astype(str),
        colorscale=[
            [0.0, COLORS["bg_primary"]],
            [0.3, COLORS["accent_dark"]],
            [0.7, COLORS["accent"]],
            [1.0, COLORS["accent_light"]],
        ],
        hovertemplate="<b>%{y} %{x}</b><br>EV Registrations: %{z:,.0f}<extra></extra>",
        colorbar=dict(
            title=dict(text="EVs", font=dict(color=COLORS["text_secondary"])),
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
    ))

    layout = {**PLOTLY_LAYOUT}
    layout.update(
        title=dict(text=f"Monthly Registration Heatmap — {county}", font=dict(size=16)),
        height=350,
        xaxis=dict(side="top", tickfont=dict(color=COLORS["text_secondary"])),
    )
    fig.update_layout(**layout)
    return fig


# ─── Feature Importance ──────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list[str]) -> go.Figure:
    """Horizontal bar chart of model feature importances."""
    importances = model.feature_importances_
    idx = np.argsort(importances)

    fig = _base_fig(
        title=dict(text="Feature Importance (Random Forest)", font=dict(size=16)),
        height=380,
    )

    clean_names = [n.replace("ev_total_", "").replace("_", " ").title() for n in feature_names]

    fig.add_trace(go.Bar(
        x=importances[idx],
        y=[clean_names[i] for i in idx],
        orientation="h",
        marker=dict(
            color=importances[idx],
            colorscale=[[0, COLORS["accent_dark"]], [1, COLORS["accent_light"]]],
            cornerradius=4,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_xaxes(title_text="Importance Score")
    fig.update_yaxes(title_text="")
    return fig


# ─── Leaderboard Table ───────────────────────────────────────────────────────

def plot_leaderboard_bar(
    summary_df: pd.DataFrame, col: str, title: str, top_n: int = 15
) -> go.Figure:
    """Horizontal bar chart for county rankings."""
    top = summary_df.nlargest(top_n, col).iloc[::-1]

    fig = _base_fig(
        title=dict(text=title, font=dict(size=16)),
        height=max(380, top_n * 30),
    )

    fig.add_trace(go.Bar(
        x=top[col],
        y=top["County"] + " (" + top["State"] + ")",
        orientation="h",
        marker=dict(
            color=top[col],
            colorscale=[[0, COLORS["accent_dark"]], [1, COLORS["accent_light"]]],
            cornerradius=4,
        ),
        hovertemplate="<b>%{y}</b><br>Value: %{x:,.1f}<extra></extra>",
    ))

    fig.update_xaxes(title_text=col.replace("_", " ").title())
    fig.update_yaxes(title_text="", tickfont=dict(size=11))
    return fig


# ─── EV Penetration Over Time ────────────────────────────────────────────────

def plot_ev_penetration(county_df: pd.DataFrame, county: str) -> go.Figure:
    """Line chart of EV penetration percentage over time."""
    fig = _base_fig(
        title=dict(text=f"EV Market Penetration — {county} County", font=dict(size=16)),
        height=360,
    )

    fig.add_trace(go.Scatter(
        x=county_df["Date"],
        y=county_df["Percent Electric Vehicles"],
        mode="lines+markers",
        line=dict(color=COLORS["accent_purple"], width=2.5),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.10)",
        name="EV %",
        hovertemplate="<b>%{x|%b %Y}</b><br>EV Share: %{y:.2f}%<extra></extra>",
    ))

    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="EV % of Total Vehicles", ticksuffix="%")
    return fig


# ─── Growth Trend Sparkline ──────────────────────────────────────────────────

def plot_growth_trend(county_df: pd.DataFrame) -> go.Figure:
    """Small sparkline-style chart for monthly EV totals."""
    fig = _base_fig(height=120, margin=dict(l=10, r=10, t=10, b=10))

    fig.add_trace(go.Scatter(
        x=county_df["Date"],
        y=county_df["Electric Vehicle (EV) Total"],
        mode="lines",
        line=dict(color=COLORS["accent"], width=2),
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.15)",
        hoverinfo="skip",
    ))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
