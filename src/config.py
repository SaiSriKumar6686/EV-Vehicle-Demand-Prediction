"""
config.py — Central configuration for paths, colors, theme, and constants.
"""
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "EV_Data.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "preprocessed_ev_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "forecasting_ev_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"
HERO_IMAGE = PROJECT_ROOT / "assets" / "ev_car.jpg"

# ─── Forecasting Defaults ────────────────────────────────────────────────────
DEFAULT_FORECAST_MONTHS = 36
MAX_FORECAST_MONTHS = 60
LAG_WINDOW = 6
MIN_COUNTY_ROWS = 6

# ─── Model Features (must match training order) ─────────────────────────────
MODEL_FEATURES = [
    "months_since_start",
    "county_encoded",
    "ev_total_lag1",
    "ev_total_lag2",
    "ev_total_lag3",
    "ev_total_roll_mean_3",
    "ev_total_pct_change_1",
    "ev_total_pct_change_3",
    "ev_growth_slope",
]

# ─── Color Palette (Gray-focused + Emerald Accent) ──────────────────────────
COLORS = {
    # Backgrounds
    "bg_primary":       "#0f1116",
    "bg_secondary":     "#181b20",
    "bg_card":          "#1f2229",
    "bg_card_alt":      "#262a33",
    "bg_elevated":      "#2d323d",
    # Borders
    "border":           "#373e4a",
    "border_subtle":    "#2a2f3a",
    # Text
    "text_primary":     "#eaeef3",
    "text_secondary":   "#8b95a5",
    "text_muted":       "#565e6c",
    # Accents
    "accent":           "#10b981",
    "accent_light":     "#34d399",
    "accent_dark":      "#059669",
    "accent_blue":      "#3b82f6",
    "accent_purple":    "#8b5cf6",
    "accent_amber":     "#f59e0b",
    "accent_red":       "#ef4444",
    "accent_teal":      "#14b8a6",
    "accent_pink":      "#ec4899",
    # Gradients
    "grad_start":       "#10b981",
    "grad_end":         "#059669",
}

# Chart color sequence (harmonious palette)
CHART_COLORS = [
    "#10b981",  # emerald
    "#6366f1",  # indigo
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#8b5cf6",  # violet
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#ec4899",  # pink
    "#3b82f6",  # blue
    "#22d3ee",  # cyan
]

# ─── Plotly Layout Template ──────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["bg_card"],
    plot_bgcolor=COLORS["bg_primary"],
    font=dict(color=COLORS["text_primary"], family="Inter, system-ui, sans-serif", size=13),
    xaxis=dict(
        gridcolor=COLORS["border_subtle"],
        zerolinecolor=COLORS["border"],
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["text_secondary"]),
    ),
    yaxis=dict(
        gridcolor=COLORS["border_subtle"],
        zerolinecolor=COLORS["border"],
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["text_secondary"]),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_secondary"]),
    ),
    margin=dict(l=60, r=30, t=50, b=50),
    hoverlabel=dict(
        bgcolor=COLORS["bg_elevated"],
        font_size=13,
        font_color=COLORS["text_primary"],
        bordercolor=COLORS["border"],
    ),
)

# ─── State Name Mappings ─────────────────────────────────────────────────────
STATE_ABBR_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "PR": "Puerto Rico",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
    "WY": "Wyoming",
}
