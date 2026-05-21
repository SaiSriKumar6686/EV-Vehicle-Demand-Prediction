"""
data_loader.py — Load data, model, and metrics with caching.
"""
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from src.config import DATA_PROCESSED, MODEL_PATH, METRICS_PATH, STATE_ABBR_TO_NAME


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """Load preprocessed EV data with type conversions and enrichments."""
    df = pd.read_csv(DATA_PROCESSED)
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["State_Name"] = df["State"].map(STATE_ABBR_TO_NAME).fillna(df["State"])
    return df


@st.cache_resource
def load_model():
    """Load the trained forecasting model (cached as resource)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(MODEL_PATH)


def load_metrics() -> dict | None:
    """Load model evaluation metrics if available."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


def get_county_list(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique counties."""
    return sorted(df["County"].dropna().unique().tolist())


def get_state_list(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique state full names."""
    return sorted(df["State_Name"].dropna().unique().tolist())


def get_counties_by_state(df: pd.DataFrame, state_name: str) -> list[str]:
    """Get sorted list of counties for a given state."""
    mask = df["State_Name"] == state_name
    return sorted(df.loc[mask, "County"].dropna().unique().tolist())


def get_county_data(df: pd.DataFrame, county: str) -> pd.DataFrame:
    """Filter and sort data for a single county."""
    return df[df["County"] == county].sort_values("Date").reset_index(drop=True)


def compute_state_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate EV totals per state per date."""
    agg = (
        df.groupby(["Date", "State", "State_Name"])
        .agg(
            BEVs=("Battery Electric Vehicles (BEVs)", "sum"),
            PHEVs=("Plug-In Hybrid Electric Vehicles (PHEVs)", "sum"),
            EV_Total=("Electric Vehicle (EV) Total", "sum"),
            Total_Vehicles=("Total Vehicles", "sum"),
        )
        .reset_index()
    )
    agg["EV_Pct"] = np.where(agg["Total_Vehicles"] > 0, agg["EV_Total"] / agg["Total_Vehicles"] * 100, 0)
    return agg.sort_values(["State", "Date"])


def compute_county_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-county summary with latest stats and growth metrics."""
    latest = df.loc[df.groupby("County")["Date"].idxmax()].copy()
    earliest = df.loc[df.groupby("County")["Date"].idxmin()].copy()

    summary = latest[["County", "State", "State_Name", "Date"]].copy()
    summary = summary.rename(columns={"Date": "Latest_Date"})
    summary["Latest_EV_Total"] = latest["Electric Vehicle (EV) Total"].values
    summary["Latest_Total_Vehicles"] = latest["Total Vehicles"].values
    summary["Latest_BEVs"] = latest["Battery Electric Vehicles (BEVs)"].values
    summary["Latest_PHEVs"] = latest["Plug-In Hybrid Electric Vehicles (PHEVs)"].values
    summary["EV_Penetration_Pct"] = np.where(
        summary["Latest_Total_Vehicles"] > 0,
        summary["Latest_EV_Total"] / summary["Latest_Total_Vehicles"] * 100,
        0,
    )

    # Compute growth from earliest to latest
    first_ev = earliest.set_index("County")["Electric Vehicle (EV) Total"]
    summary = summary.set_index("County")
    summary["First_EV_Total"] = first_ev
    summary["Growth_Abs"] = summary["Latest_EV_Total"] - summary["First_EV_Total"]
    summary["Growth_Pct"] = np.where(
        summary["First_EV_Total"] > 0,
        (summary["Growth_Abs"] / summary["First_EV_Total"]) * 100,
        0,
    )
    summary = summary.reset_index()

    # Count data points per county
    county_counts = df.groupby("County").size().rename("Data_Points")
    summary = summary.merge(county_counts, on="County", how="left")

    return summary.sort_values("Latest_EV_Total", ascending=False).reset_index(drop=True)
