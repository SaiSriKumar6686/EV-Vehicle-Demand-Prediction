"""
forecaster.py — Forecasting logic with confidence intervals.
All forecasting code lives here to avoid duplication.
"""
import pandas as pd
import numpy as np
from src.config import MODEL_FEATURES, LAG_WINDOW


def _safe_lags(history: list[float]) -> tuple[float, float, float]:
    """Extract lag1, lag2, lag3 from history, padding with 0 if short."""
    n = len(history)
    return (
        history[-1] if n >= 1 else 0.0,
        history[-2] if n >= 2 else 0.0,
        history[-3] if n >= 3 else 0.0,
    )


def _build_feature_row(
    months_since_start: int,
    county_code: int,
    history: list[float],
    cumulative: list[float],
) -> dict:
    """Construct a single feature row for the model."""
    lag1, lag2, lag3 = _safe_lags(history)
    roll_mean = np.mean([lag1, lag2, lag3])
    pct1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0.0
    pct3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0.0

    recent_cum = cumulative[-LAG_WINDOW:]
    slope = (
        np.polyfit(range(len(recent_cum)), recent_cum, 1)[0]
        if len(recent_cum) >= 2
        else 0.0
    )

    return {
        "months_since_start": months_since_start,
        "county_encoded": county_code,
        "ev_total_lag1": lag1,
        "ev_total_lag2": lag2,
        "ev_total_lag3": lag3,
        "ev_total_roll_mean_3": roll_mean,
        "ev_total_pct_change_1": pct1,
        "ev_total_pct_change_3": pct3,
        "ev_growth_slope": slope,
    }


def generate_forecast(
    model,
    county_df: pd.DataFrame,
    horizon_months: int = 36,
) -> pd.DataFrame:
    """
    Run autoregressive forecast for a single county.

    Returns DataFrame with columns:
        Date, Predicted_EV_Total, Cumulative_EV, Lower_CI, Upper_CI
    """
    historical_ev = list(county_df["Electric Vehicle (EV) Total"].values[-LAG_WINDOW:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since = county_df["months_since_start"].max()
    county_code = county_df["county_encoded"].iloc[0]
    latest_date = county_df["Date"].max()

    rows = []
    for i in range(1, horizon_months + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since += 1

        features = _build_feature_row(months_since, county_code, historical_ev, cumulative_ev)
        X = pd.DataFrame([features])[MODEL_FEATURES].values  # numpy array avoids feature-name warnings

        # Point prediction
        pred = max(0, model.predict(X)[0])

        # Confidence interval from individual tree predictions
        if hasattr(model, "estimators_"):
            tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
            lower = max(0, float(np.percentile(tree_preds, 10)))
            upper = max(0, float(np.percentile(tree_preds, 90)))
        else:
            lower = pred * 0.8
            upper = pred * 1.2

        rows.append({
            "Date": forecast_date,
            "Predicted_EV_Total": round(pred),
            "Lower_CI": round(lower),
            "Upper_CI": round(upper),
        })

        # Roll forward
        historical_ev.append(pred)
        if len(historical_ev) > LAG_WINDOW:
            historical_ev.pop(0)
        cumulative_ev.append(cumulative_ev[-1] + pred)
        if len(cumulative_ev) > LAG_WINDOW:
            cumulative_ev.pop(0)

    forecast_df = pd.DataFrame(rows)

    # Build cumulative columns
    hist_cumulative_total = county_df["Electric Vehicle (EV) Total"].sum()
    forecast_df["Cumulative_EV"] = forecast_df["Predicted_EV_Total"].cumsum() + hist_cumulative_total
    forecast_df["Cumulative_Lower"] = forecast_df["Lower_CI"].cumsum() + hist_cumulative_total
    forecast_df["Cumulative_Upper"] = forecast_df["Upper_CI"].cumsum() + hist_cumulative_total

    return forecast_df


def build_historical_cumulative(county_df: pd.DataFrame) -> pd.DataFrame:
    """Build historical cumulative EV data for plotting."""
    hist = county_df[["Date", "Electric Vehicle (EV) Total",
                       "Battery Electric Vehicles (BEVs)",
                       "Plug-In Hybrid Electric Vehicles (PHEVs)"]].copy()
    hist = hist.rename(columns={"Electric Vehicle (EV) Total": "EV_Total"})
    hist["Cumulative_EV"] = hist["EV_Total"].cumsum()
    hist["Cumulative_BEV"] = hist["Battery Electric Vehicles (BEVs)"].cumsum()
    hist["Cumulative_PHEV"] = hist["Plug-In Hybrid Electric Vehicles (PHEVs)"].cumsum()
    hist["Source"] = "Historical"
    return hist


def compute_growth_metrics(
    county_df: pd.DataFrame, forecast_df: pd.DataFrame
) -> dict:
    """Compute growth summary metrics."""
    hist_total = county_df["Electric Vehicle (EV) Total"].sum()
    forecast_total = forecast_df["Predicted_EV_Total"].sum()
    combined = hist_total + forecast_total

    growth_pct = ((forecast_total / hist_total) * 100) if hist_total > 0 else 0.0
    avg_monthly = forecast_total / len(forecast_df) if len(forecast_df) > 0 else 0.0

    latest_monthly = county_df["Electric Vehicle (EV) Total"].iloc[-1] if len(county_df) > 0 else 0
    forecast_latest = forecast_df["Predicted_EV_Total"].iloc[-1] if len(forecast_df) > 0 else 0

    return {
        "historical_total": hist_total,
        "forecast_total": forecast_total,
        "combined_total": combined,
        "growth_pct": growth_pct,
        "avg_monthly_forecast": avg_monthly,
        "latest_historical_monthly": latest_monthly,
        "last_forecast_monthly": forecast_latest,
        "trend": "increasing" if forecast_latest > latest_monthly else "stable/decreasing",
    }
