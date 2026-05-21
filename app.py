"""
EV Demand Forecaster — Production Streamlit Dashboard
=====================================================
A real-world deployable tool for forecasting Electric Vehicle adoption
across US counties using a Random Forest model with autoregressive features.
"""
import streamlit as st
import pandas as pd
import numpy as np
import io

# ─── Page Config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="EV Demand Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import COLORS, CHART_COLORS, MODEL_FEATURES, DEFAULT_FORECAST_MONTHS, MAX_FORECAST_MONTHS
from src.data_loader import (
    load_data, load_model, load_metrics,
    get_county_list, get_state_list, get_counties_by_state,
    get_county_data, compute_state_aggregates, compute_county_summary,
)
from src.forecaster import generate_forecast, build_historical_cumulative, compute_growth_metrics
from src.visualizations import (
    plot_forecast, plot_monthly_forecast, plot_bev_phev_split,
    plot_comparison, plot_state_map, plot_monthly_heatmap,
    plot_feature_importance, plot_leaderboard_bar,
    plot_ev_penetration, plot_growth_trend,
)


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Gray-focused palette with emerald accent
# ═════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Root & Global ─────────────────────────────────────── */
        html, body, .stApp {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: {COLORS['bg_primary']};
            color: {COLORS['text_primary']};
        }}
        .stApp {{
            background: linear-gradient(180deg, {COLORS['bg_primary']} 0%, #0a0d12 100%);
        }}

        /* ── Sidebar ───────────────────────────────────────────── */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['bg_secondary']} 0%, #12151a 100%);
            border-right: 1px solid {COLORS['border_subtle']};
        }}
        section[data-testid="stSidebar"] .stRadio > label {{
            color: {COLORS['text_secondary']} !important;
            font-weight: 500;
            font-size: 13px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
            padding: 10px 16px;
            margin: 2px 0;
            border-radius: 10px;
            transition: all 0.2s ease;
            color: {COLORS['text_secondary']};
            font-size: 14px;
        }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
            background: {COLORS['bg_card']};
            color: {COLORS['text_primary']};
        }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] {{
            background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
            color: {COLORS['accent_light']};
            border-left: 3px solid {COLORS['accent']};
        }}

        /* ── Headers ───────────────────────────────────────────── */
        h1 {{ color: {COLORS['text_primary']}; font-weight: 700; letter-spacing: -0.5px; }}
        h2 {{ color: {COLORS['text_primary']}; font-weight: 600; }}
        h3 {{ color: {COLORS['text_secondary']}; font-weight: 500; font-size: 16px; text-transform: uppercase; letter-spacing: 0.5px; }}

        /* ── Metric Cards ──────────────────────────────────────── */
        [data-testid="stMetric"] {{
            background: linear-gradient(135deg, {COLORS['bg_card']} 0%, {COLORS['bg_card_alt']} 100%);
            border: 1px solid {COLORS['border_subtle']};
            border-radius: 14px;
            padding: 18px 22px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }}
        [data-testid="stMetric"]:hover {{
            border-color: {COLORS['accent']};
            box-shadow: 0 4px 25px rgba(16,185,129,0.1);
            transform: translateY(-2px);
        }}
        [data-testid="stMetricLabel"] {{
            color: {COLORS['text_secondary']} !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.7px;
        }}
        [data-testid="stMetricValue"] {{
            color: {COLORS['text_primary']} !important;
            font-weight: 700 !important;
            font-size: 28px !important;
        }}
        [data-testid="stMetricDelta"] {{
            font-size: 13px !important;
        }}

        /* ── Selectbox / Input ─────────────────────────────────── */
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stNumberInput > div > div > input {{
            background-color: {COLORS['bg_card']} !important;
            border: 1px solid {COLORS['border']} !important;
            border-radius: 10px !important;
            color: {COLORS['text_primary']} !important;
            transition: border-color 0.2s ease;
        }}
        .stSelectbox > div > div:focus-within,
        .stMultiSelect > div > div:focus-within {{
            border-color: {COLORS['accent']} !important;
            box-shadow: 0 0 0 2px rgba(16,185,129,0.15) !important;
        }}

        /* ── Tabs ──────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            background: {COLORS['bg_secondary']};
            border-radius: 12px;
            padding: 4px;
            border: 1px solid {COLORS['border_subtle']};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px;
            padding: 10px 20px;
            color: {COLORS['text_secondary']};
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: {COLORS['text_primary']};
            background: {COLORS['bg_card']};
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {COLORS['accent_dark']}, {COLORS['accent']}) !important;
            color: white !important;
            font-weight: 600;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            display: none;
        }}
        .stTabs [data-baseweb="tab-border"] {{
            display: none;
        }}

        /* ── Buttons ───────────────────────────────────────────── */
        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['accent_dark']}, {COLORS['accent']});
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 24px;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(16,185,129,0.2);
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16,185,129,0.3);
        }}

        /* ── Download buttons ──────────────────────────────────── */
        .stDownloadButton > button {{
            background: {COLORS['bg_card']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        .stDownloadButton > button:hover {{
            border-color: {COLORS['accent']};
            background: {COLORS['bg_card_alt']};
        }}

        /* ── Expander ──────────────────────────────────────────── */
        .streamlit-expanderHeader {{
            background-color: {COLORS['bg_card']};
            border: 1px solid {COLORS['border_subtle']};
            border-radius: 10px;
            color: {COLORS['text_primary']};
            font-weight: 500;
        }}

        /* ── Dataframes ────────────────────────────────────────── */
        .stDataFrame {{
            border: 1px solid {COLORS['border_subtle']};
            border-radius: 12px;
            overflow: hidden;
        }}

        /* ── Success / Info / Warning blocks ───────────────────── */
        .stAlert {{
            border-radius: 12px;
            border: none;
        }}

        /* ── Plotly chart containers ───────────────────────────── */
        .stPlotlyChart {{
            border: 1px solid {COLORS['border_subtle']};
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}

        /* ── Dividers ──────────────────────────────────────────── */
        hr {{
            border-color: {COLORS['border_subtle']};
            opacity: 0.5;
        }}

        /* ── Custom card class ─────────────────────────────────── */
        .glass-card {{
            background: linear-gradient(135deg, {COLORS['bg_card']} 0%, {COLORS['bg_card_alt']} 100%);
            border: 1px solid {COLORS['border_subtle']};
            border-radius: 16px;
            padding: 24px;
            margin: 8px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        .hero-title {{
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(135deg, {COLORS['accent_light']}, {COLORS['accent_blue']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -1px;
            line-height: 1.1;
        }}
        .hero-sub {{
            font-size: 18px;
            color: {COLORS['text_secondary']};
            font-weight: 400;
            margin-top: 8px;
        }}
        .stat-label {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {COLORS['text_muted']};
            margin-bottom: 4px;
        }}
        .stat-value {{
            font-size: 30px;
            font-weight: 700;
            color: {COLORS['text_primary']};
        }}
        .accent-text {{ color: {COLORS['accent']}; }}
        .muted-text {{ color: {COLORS['text_muted']}; }}
        .section-header {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: {COLORS['text_muted']};
            padding: 16px 0 8px 0;
            border-bottom: 1px solid {COLORS['border_subtle']};
            margin-bottom: 16px;
        }}
    </style>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ═════════════════════════════════════════════════════════════════════════════
inject_css()
df = load_data()
model = load_model()
metrics = load_metrics()

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <span style="font-size: 36px;">⚡</span>
            <div style="font-size: 18px; font-weight: 700; color: {COLORS['text_primary']};
                        margin-top: 4px; letter-spacing: -0.5px;">EV Forecaster</div>
            <div style="font-size: 11px; color: {COLORS['text_muted']}; text-transform: uppercase;
                        letter-spacing: 1.5px; margin-top: 2px;">Demand Prediction</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Dashboard", "🔮  Forecast", "📊  Compare", "📈  Analytics", "🤖  Model", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
        <div style="text-align:center; padding: 10px 0; color: {COLORS['text_muted']}; font-size: 11px;">
            Data: {df['Date'].min().strftime('%b %Y')} — {df['Date'].max().strftime('%b %Y')}<br>
            {df['County'].nunique()} Counties · {df['State'].nunique()} States
        </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown("""
        <div style="padding: 40px 0 20px 0; text-align: center;">
            <div class="hero-title">EV Demand Forecaster</div>
            <div class="hero-sub">AI-powered Electric Vehicle adoption forecasting across US counties</div>
        </div>
    """, unsafe_allow_html=True)

    # KPI Row
    total_evs = df["Electric Vehicle (EV) Total"].sum()
    total_bevs = df["Battery Electric Vehicles (BEVs)"].sum()
    total_phevs = df["Plug-In Hybrid Electric Vehicles (PHEVs)"].sum()
    avg_penetration = df["Percent Electric Vehicles"].mean()
    n_counties = df["County"].nunique()
    n_states = df["State"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total EV Registrations", f"{total_evs:,.0f}")
    c2.metric("Battery EVs (BEVs)", f"{total_bevs:,.0f}")
    c3.metric("Plug-in Hybrids", f"{total_phevs:,.0f}")
    c4.metric("Counties Tracked", f"{n_counties}")
    c5.metric("States & Territories", f"{n_states}")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # State Map + Top Counties
    col_map, col_top = st.columns([3, 2])

    with col_map:
        state_agg = compute_state_aggregates(df)
        fig_map = plot_state_map(state_agg, "EV_Total", "EV Registrations by State")
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    with col_top:
        st.markdown('<div class="section-header">Top Counties by EV Registrations</div>', unsafe_allow_html=True)
        summary = compute_county_summary(df)
        top_10 = summary.head(10)[["County", "State", "Latest_EV_Total", "EV_Penetration_Pct"]].reset_index(drop=True)
        top_10.columns = ["County", "State", "Latest EVs", "EV %"]
        top_10["EV %"] = top_10["EV %"].map(lambda x: f"{x:.2f}%")
        top_10.index = top_10.index + 1
        st.dataframe(top_10, use_container_width=True, height=380)

    # Quick Insights Row
    st.markdown('<div class="section-header">Quick Insights</div>', unsafe_allow_html=True)
    ci1, ci2, ci3 = st.columns(3)

    bev_share = (total_bevs / total_evs * 100) if total_evs > 0 else 0
    ci1.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="stat-label">BEV Market Share</div>
            <div class="stat-value accent-text">{bev_share:.1f}%</div>
            <div class="muted-text" style="font-size:12px; margin-top:4px;">Battery EVs dominate the mix</div>
        </div>
    """, unsafe_allow_html=True)

    latest_month = df["Date"].max().strftime("%B %Y")
    latest_total = df[df["Date"] == df["Date"].max()]["Electric Vehicle (EV) Total"].sum()
    ci2.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="stat-label">Latest Month ({latest_month})</div>
            <div class="stat-value">{latest_total:,.0f}</div>
            <div class="muted-text" style="font-size:12px; margin-top:4px;">EV registrations recorded</div>
        </div>
    """, unsafe_allow_html=True)

    ci3.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="stat-label">Avg EV Penetration</div>
            <div class="stat-value" style="color:{COLORS['accent_purple']};">{avg_penetration:.2f}%</div>
            <div class="muted-text" style="font-size:12px; margin-top:4px;">of total registered vehicles</div>
        </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: FORECAST
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Forecast":
    st.markdown("""
        <div style="padding: 20px 0 10px 0;">
            <div class="hero-title" style="font-size:32px;">County EV Forecast</div>
            <div class="hero-sub" style="font-size:15px;">Select a state and county to generate a detailed EV adoption forecast</div>
        </div>
    """, unsafe_allow_html=True)

    # Filters row
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        states = get_state_list(df)
        selected_state = st.selectbox("State", states, index=states.index("Washington") if "Washington" in states else 0)
    with f2:
        counties = get_counties_by_state(df, selected_state)
        if counties:
            selected_county = st.selectbox("County", counties)
        else:
            st.warning("No counties found for this state.")
            st.stop()
    with f3:
        horizon = st.number_input("Forecast Months", min_value=6, max_value=MAX_FORECAST_MONTHS, value=DEFAULT_FORECAST_MONTHS, step=6)

    county_df = get_county_data(df, selected_county)

    if len(county_df) < 3:
        st.warning(f"Insufficient data for {selected_county} (only {len(county_df)} data points). Select another county.")
        st.stop()

    # Run forecast
    forecast_df = generate_forecast(model, county_df, horizon_months=horizon)
    historical = build_historical_cumulative(county_df)
    growth = compute_growth_metrics(county_df, forecast_df)

    # Growth metrics row
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Historical Total", f"{growth['historical_total']:,.0f}")
    m2.metric("Forecast Total", f"{growth['forecast_total']:,.0f}")
    m3.metric("Projected Growth", f"{growth['growth_pct']:.1f}%",
              delta="↑ Increasing" if growth['trend'] == "increasing" else "→ Stable")
    m4.metric("Avg Monthly (Forecast)", f"{growth['avg_monthly_forecast']:.1f}")

    # Main forecast chart
    horizon_label = f"{horizon // 12} Year{'s' if horizon > 12 else ''}" if horizon >= 12 else f"{horizon} Months"
    fig_forecast = plot_forecast(historical, forecast_df, selected_county, horizon_label)
    st.plotly_chart(fig_forecast, use_container_width=True, config={"displayModeBar": False})

    # Sub-charts in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Monthly Forecast", "🔋 BEV vs PHEV", "🌡️ Heatmap", "📈 Market Penetration"])

    with tab1:
        fig_monthly = plot_monthly_forecast(forecast_df, selected_county)
        st.plotly_chart(fig_monthly, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        col_d, col_info = st.columns([2, 1])
        with col_d:
            fig_donut = plot_bev_phev_split(county_df, selected_county)
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        with col_info:
            total_bev = county_df["Battery Electric Vehicles (BEVs)"].sum()
            total_phev = county_df["Plug-In Hybrid Electric Vehicles (PHEVs)"].sum()
            total = total_bev + total_phev
            st.markdown(f"""
                <div class="glass-card">
                    <div class="stat-label">Battery EVs (BEV)</div>
                    <div style="font-size:24px; font-weight:700; color:{COLORS['accent']};">{total_bev:,.0f}</div>
                    <div class="muted-text">{(total_bev/total*100) if total > 0 else 0:.1f}% of total</div>
                    <div style="height:16px;"></div>
                    <div class="stat-label">Plug-in Hybrids (PHEV)</div>
                    <div style="font-size:24px; font-weight:700; color:{COLORS['accent_purple']};">{total_phev:,.0f}</div>
                    <div class="muted-text">{(total_phev/total*100) if total > 0 else 0:.1f}% of total</div>
                </div>
            """, unsafe_allow_html=True)

    with tab3:
        fig_heat = plot_monthly_heatmap(county_df, selected_county)
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    with tab4:
        fig_pen = plot_ev_penetration(county_df, selected_county)
        st.plotly_chart(fig_pen, use_container_width=True, config={"displayModeBar": False})

    # Download Section
    st.markdown("---")
    st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        csv_forecast = forecast_df.to_csv(index=False)
        st.download_button("📥 Forecast Data (CSV)", csv_forecast,
                           f"{selected_county}_forecast.csv", "text/csv")
    with dl2:
        csv_hist = county_df.to_csv(index=False)
        st.download_button("📥 Historical Data (CSV)", csv_hist,
                           f"{selected_county}_historical.csv", "text/csv")
    with dl3:
        # Combined Excel
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            county_df.to_excel(writer, sheet_name="Historical", index=False)
            forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
        st.download_button("📥 Full Report (Excel)", buf.getvalue(),
                           f"{selected_county}_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: COMPARE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊  Compare":
    st.markdown("""
        <div style="padding: 20px 0 10px 0;">
            <div class="hero-title" style="font-size:32px;">Compare Counties</div>
            <div class="hero-sub" style="font-size:15px;">Overlay EV adoption trends for up to 5 counties side by side</div>
        </div>
    """, unsafe_allow_html=True)

    county_list = get_county_list(df)

    fc1, fc2 = st.columns([3, 1])
    with fc1:
        multi_counties = st.multiselect(
            "Select counties to compare",
            county_list,
            max_selections=5,
            placeholder="Choose up to 5 counties...",
        )
    with fc2:
        cmp_horizon = st.number_input("Forecast months", min_value=6, max_value=MAX_FORECAST_MONTHS, value=36, step=6, key="cmp_horizon")

    if multi_counties:
        comparison_data = []
        growth_rows = []

        for cty in multi_counties:
            cty_df = get_county_data(df, cty)
            if len(cty_df) < 3:
                st.warning(f"Skipping {cty} — insufficient data ({len(cty_df)} rows).")
                continue

            fc_df = generate_forecast(model, cty_df, cmp_horizon)
            hist = build_historical_cumulative(cty_df)
            gm = compute_growth_metrics(cty_df, fc_df)

            comparison_data.append({
                "county": cty,
                "hist_dates": hist["Date"].tolist(),
                "hist_cum": hist["Cumulative_EV"].tolist(),
                "fc_dates": fc_df["Date"].tolist(),
                "fc_cum": fc_df["Cumulative_EV"].tolist(),
            })

            growth_rows.append({
                "County": cty,
                "State": cty_df["State"].iloc[0],
                "Historical Total": int(gm["historical_total"]),
                "Forecast Total": int(gm["forecast_total"]),
                "Growth %": f"{gm['growth_pct']:.1f}%",
                "Trend": "📈" if gm["trend"] == "increasing" else "➡️",
            })

        if comparison_data:
            fig_cmp = plot_comparison(comparison_data)
            st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="section-header">Growth Comparison</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(growth_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Select at least one county above to start comparing.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈  Analytics":
    st.markdown("""
        <div style="padding: 20px 0 10px 0;">
            <div class="hero-title" style="font-size:32px;">Analytics & Insights</div>
            <div class="hero-sub" style="font-size:15px;">Explore EV adoption trends across the United States</div>
        </div>
    """, unsafe_allow_html=True)

    summary = compute_county_summary(df)
    state_agg = compute_state_aggregates(df)

    tab_lb, tab_state, tab_time = st.tabs(["🏆 Leaderboard", "🗺️ State Analysis", "📅 Time Trends"])

    with tab_lb:
        lb1, lb2 = st.columns(2)
        with lb1:
            fig_top_ev = plot_leaderboard_bar(summary, "Latest_EV_Total", "Top 15 Counties — Latest EV Count", 15)
            st.plotly_chart(fig_top_ev, use_container_width=True, config={"displayModeBar": False})
        with lb2:
            # Filter counties with meaningful penetration
            pen_df = summary[summary["EV_Penetration_Pct"] > 0].copy()
            fig_top_pen = plot_leaderboard_bar(pen_df, "EV_Penetration_Pct", "Top 15 Counties — EV Penetration %", 15)
            st.plotly_chart(fig_top_pen, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-header">Full County Rankings</div>', unsafe_allow_html=True)
        display_df = summary[["County", "State", "Latest_EV_Total", "Latest_BEVs", "Latest_PHEVs",
                              "EV_Penetration_Pct", "Growth_Abs", "Data_Points"]].copy()
        display_df.columns = ["County", "State", "Latest EVs", "BEVs", "PHEVs", "EV %", "Growth", "Data Pts"]
        display_df["EV %"] = display_df["EV %"].map(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)

    with tab_state:
        sm1, sm2 = st.columns(2)
        with sm1:
            fig_map_ev = plot_state_map(state_agg, "EV_Total", "Total EV Registrations by State")
            st.plotly_chart(fig_map_ev, use_container_width=True, config={"displayModeBar": False})
        with sm2:
            fig_map_pct = plot_state_map(state_agg, "EV_Pct", "EV Penetration % by State")
            st.plotly_chart(fig_map_pct, use_container_width=True, config={"displayModeBar": False})

        # State-level aggregated table
        st.markdown('<div class="section-header">State-Level Summary</div>', unsafe_allow_html=True)
        latest_state = state_agg.loc[state_agg.groupby("State")["Date"].idxmax()].copy()
        latest_state = latest_state.sort_values("EV_Total", ascending=False)
        state_display = latest_state[["State_Name", "State", "EV_Total", "BEVs", "PHEVs", "EV_Pct"]].copy()
        state_display.columns = ["State", "Code", "Total EVs", "BEVs", "PHEVs", "EV %"]
        state_display["EV %"] = state_display["EV %"].map(lambda x: f"{x:.2f}%")
        st.dataframe(state_display, use_container_width=True, height=400, hide_index=True)

    with tab_time:
        # National monthly trend
        monthly_national = df.groupby("Date").agg(
            EV_Total=("Electric Vehicle (EV) Total", "sum"),
            BEVs=("Battery Electric Vehicles (BEVs)", "sum"),
            PHEVs=("Plug-In Hybrid Electric Vehicles (PHEVs)", "sum"),
        ).reset_index().sort_values("Date")

        import plotly.graph_objects as go
        from src.config import PLOTLY_LAYOUT

        fig_trend = go.Figure()
        fig_trend.update_layout(**PLOTLY_LAYOUT, title=dict(text="National Monthly EV Registrations", font=dict(size=18)), height=420)
        fig_trend.add_trace(go.Scatter(
            x=monthly_national["Date"], y=monthly_national["EV_Total"],
            mode="lines", name="Total EVs",
            line=dict(color=COLORS["accent"], width=2.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Total: %{y:,.0f}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_national["Date"], y=monthly_national["BEVs"],
            mode="lines", name="BEVs",
            line=dict(color=COLORS["accent_blue"], width=2, dash="dot"),
            hovertemplate="<b>%{x|%b %Y}</b><br>BEVs: %{y:,.0f}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_national["Date"], y=monthly_national["PHEVs"],
            mode="lines", name="PHEVs",
            line=dict(color=COLORS["accent_purple"], width=2, dash="dot"),
            hovertemplate="<b>%{x|%b %Y}</b><br>PHEVs: %{y:,.0f}<extra></extra>",
        ))
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

        # Year-over-year comparison
        yoy = df.groupby(["year", "month"]).agg(
            EV_Total=("Electric Vehicle (EV) Total", "sum")
        ).reset_index()
        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        yoy["Month_Name"] = yoy["month"].map(month_names)

        fig_yoy = go.Figure()
        fig_yoy.update_layout(**PLOTLY_LAYOUT, title=dict(text="Year-over-Year Monthly Comparison", font=dict(size=18)), height=420)
        for i, yr in enumerate(sorted(yoy["year"].unique())):
            yr_data = yoy[yoy["year"] == yr]
            fig_yoy.add_trace(go.Bar(
                x=yr_data["Month_Name"], y=yr_data["EV_Total"],
                name=str(yr),
                marker=dict(color=CHART_COLORS[i % len(CHART_COLORS)], cornerradius=4),
                hovertemplate=f"<b>{yr}</b> — " + "%{x}<br>EVs: %{y:,.0f}<extra></extra>",
            ))
        fig_yoy.update_layout(barmode="group")
        st.plotly_chart(fig_yoy, use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model":
    st.markdown("""
        <div style="padding: 20px 0 10px 0;">
            <div class="hero-title" style="font-size:32px;">Model Information</div>
            <div class="hero-sub" style="font-size:15px;">Transparency into the machine learning model powering the forecasts</div>
        </div>
    """, unsafe_allow_html=True)

    tab_perf, tab_fi, tab_method = st.tabs(["📊 Performance", "🎯 Feature Importance", "📘 Methodology"])

    with tab_perf:
        if metrics:
            st.markdown('<div class="section-header">Model Evaluation Metrics</div>', unsafe_allow_html=True)

            p1, p2, p3, p4 = st.columns(4)
            test = metrics.get("test_metrics", {})
            train = metrics.get("train_metrics", {})

            p1.metric("Test R² Score", f"{test.get('r2', 'N/A')}")
            p2.metric("Test MAE", f"{test.get('mae', 'N/A')}")
            p3.metric("Test RMSE", f"{test.get('rmse', 'N/A')}")
            p4.metric("CV R² (5-fold)", f"{metrics.get('cv_r2_mean', 'N/A')} ± {metrics.get('cv_r2_std', 'N/A')}")

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"""
                    <div class="glass-card">
                        <div class="stat-label">Training Set</div>
                        <div style="font-size:18px; font-weight:600; color:{COLORS['text_primary']}; margin:8px 0;">
                            {metrics.get('train_size', 'N/A'):,} samples
                        </div>
                        <div style="display:flex; gap:24px; margin-top:12px;">
                            <div><span class="muted-text">R²:</span> <span style="color:{COLORS['accent']};">{train.get('r2','N/A')}</span></div>
                            <div><span class="muted-text">MAE:</span> <span>{train.get('mae','N/A')}</span></div>
                            <div><span class="muted-text">RMSE:</span> <span>{train.get('rmse','N/A')}</span></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with tc2:
                st.markdown(f"""
                    <div class="glass-card">
                        <div class="stat-label">Test Set</div>
                        <div style="font-size:18px; font-weight:600; color:{COLORS['text_primary']}; margin:8px 0;">
                            {metrics.get('test_size', 'N/A'):,} samples
                        </div>
                        <div style="display:flex; gap:24px; margin-top:12px;">
                            <div><span class="muted-text">R²:</span> <span style="color:{COLORS['accent']};">{test.get('r2','N/A')}</span></div>
                            <div><span class="muted-text">MAE:</span> <span>{test.get('mae','N/A')}</span></div>
                            <div><span class="muted-text">RMSE:</span> <span>{test.get('rmse','N/A')}</span></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No evaluation metrics found. Run `python scripts/train_model.py` to generate metrics.")

            # Show model params from the model object itself
            st.markdown('<div class="section-header">Model Parameters</div>', unsafe_allow_html=True)
            st.json(model.get_params())

    with tab_fi:
        fig_fi = plot_feature_importance(model, MODEL_FEATURES)
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

        if metrics and "feature_importances" in metrics:
            st.markdown('<div class="section-header">Importance Scores</div>', unsafe_allow_html=True)
            fi_df = pd.DataFrame([
                {"Feature": k.replace("ev_total_", "").replace("_", " ").title(), "Importance": v}
                for k, v in metrics["feature_importances"].items()
            ]).sort_values("Importance", ascending=False)
            fi_df["Importance"] = fi_df["Importance"].map(lambda x: f"{x:.4f}")
            st.dataframe(fi_df, use_container_width=True, hide_index=True)

    with tab_method:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color:{COLORS['accent_light']}; margin-top:0;">Algorithm</h3>
            <p><strong>Random Forest Regressor</strong> — an ensemble of 200 decision trees, each trained on a bootstrapped
            subset of the data. Predictions are averaged across all trees, providing robust estimates and
            natural uncertainty quantification.</p>

            <h3 style="color:{COLORS['accent_light']};">Feature Engineering</h3>
            <ul>
                <li><strong>Lag Features</strong> (lag1, lag2, lag3): Past 3 months of EV registrations capture recent momentum</li>
                <li><strong>Rolling Mean</strong>: 3-month moving average smooths short-term noise</li>
                <li><strong>Percentage Changes</strong>: 1-month and 3-month growth rates capture acceleration/deceleration</li>
                <li><strong>Growth Slope</strong>: Linear regression slope over recent cumulative values captures long-term trend</li>
                <li><strong>County Encoding</strong>: Label-encoded county identifier captures county-specific baseline adoption levels</li>
                <li><strong>Temporal Feature</strong>: Months since start of data captures time progression</li>
            </ul>

            <h3 style="color:{COLORS['accent_light']};">Forecasting Method</h3>
            <p><strong>Autoregressive rollforward:</strong> Each monthly prediction feeds back into the feature
            calculations for the next month, creating a chain of dependent predictions. This naturally
            captures compounding growth effects.</p>

            <h3 style="color:{COLORS['accent_light']};">Confidence Intervals</h3>
            <p>The 80% confidence band is derived from the 10th and 90th percentiles of individual
            tree predictions within the Random Forest ensemble, providing a natural measure of
            prediction uncertainty without requiring additional bootstrapping.</p>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown("""
        <div style="padding: 20px 0 10px 0;">
            <div class="hero-title" style="font-size:32px;">About This Project</div>
            <div class="hero-sub" style="font-size:15px;">EV Demand Forecaster — Built for real-world deployment</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass-card">
        <h3 style="color:{COLORS['accent_light']}; margin-top:0;">📋 Project Overview</h3>
        <p>This application forecasts Electric Vehicle (EV) adoption across US counties using
        machine learning. It ingests historical registration data, engineers temporal features,
        and uses a Random Forest model to generate multi-year forecasts with confidence intervals.</p>

        <h3 style="color:{COLORS['accent_light']};">📊 Data Source</h3>
        <p>US EV registration data spanning <strong>{df['Date'].min().strftime('%B %Y')}</strong> to
        <strong>{df['Date'].max().strftime('%B %Y')}</strong>, covering
        <strong>{df['County'].nunique()}</strong> counties across
        <strong>{df['State'].nunique()}</strong> states and territories.
        The dataset includes both Battery Electric Vehicles (BEVs) and
        Plug-in Hybrid Electric Vehicles (PHEVs).</p>

        <h3 style="color:{COLORS['accent_light']};">🛠️ Technology Stack</h3>
        <ul>
            <li><strong>Frontend:</strong> Streamlit with custom CSS theming</li>
            <li><strong>ML Model:</strong> scikit-learn RandomForestRegressor</li>
            <li><strong>Visualization:</strong> Plotly (interactive charts)</li>
            <li><strong>Data Processing:</strong> pandas, NumPy</li>
            <li><strong>Deployment:</strong> Docker-ready, Streamlit Cloud compatible</li>
        </ul>

        <h3 style="color:{COLORS['accent_light']};">👤 Author</h3>
        <p>Developed by <strong>SaiSriKumar Parimi</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Data coverage stats
    st.markdown('<div class="section-header">Data Coverage</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Records", f"{len(df):,}")
    d2.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days // 30} months")
    d3.metric("Vehicle Types", "Passenger & Truck")
    d4.metric("Model Features", f"{len(MODEL_FEATURES)}")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(f"""
    <div style="text-align:center; padding: 40px 0 20px 0; border-top: 1px solid {COLORS['border_subtle']}; margin-top: 40px;">
        <span style="font-size:11px; color:{COLORS['text_muted']}; letter-spacing: 1px;">
            EV DEMAND FORECASTER · Built by SaiSriKumar Parimi · Powered by Streamlit & scikit-learn
        </span>
    </div>
""", unsafe_allow_html=True)
