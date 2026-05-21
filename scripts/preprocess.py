"""
preprocess.py — Transform raw EV_Data.csv into preprocessed_ev_data.csv.

Usage:
    python scripts/preprocess.py

This script:
  1. Parses and normalizes date strings
  2. Cleans numeric columns (removes commas)
  3. Label-encodes counties
  4. Computes lag features, rolling stats, and growth metrics per county
  5. Drops counties with insufficient history
  6. Saves to data/processed/preprocessed_ev_data.csv
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def main():
    raw_path = PROJECT_ROOT / "data" / "raw" / "EV_Data.csv"
    out_path = PROJECT_ROOT / "data" / "processed" / "preprocessed_ev_data.csv"

    print(f"📂 Reading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"   Raw shape: {df.shape}")

    # ── 1. Parse dates ────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)
    df = df.sort_values(["County", "State", "Vehicle Primary Use", "Date"]).reset_index(drop=True)

    # ── 2. Clean numeric columns (remove commas) ─────────────────────────
    numeric_cols = [
        "Battery Electric Vehicles (BEVs)",
        "Plug-In Hybrid Electric Vehicles (PHEVs)",
        "Electric Vehicle (EV) Total",
        "Non-Electric Vehicle Total",
        "Total Vehicles",
        "Percent Electric Vehicles",
    ]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    # ── 3. Temporal features ──────────────────────────────────────────────
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["numeric_date"] = df["year"] * 12 + df["month"]

    # ── 4. Label-encode counties ──────────────────────────────────────────
    county_labels = sorted(df["County"].unique())
    county_map = {c: i for i, c in enumerate(county_labels)}
    df["county_encoded"] = df["County"].map(county_map)

    # ── 5. Per-county group features ──────────────────────────────────────
    # Group by county + vehicle use to maintain separate time series
    group_cols = ["County", "State", "Vehicle Primary Use"]
    ev_col = "Electric Vehicle (EV) Total"

    # months_since_start: per-group sequence index
    df["months_since_start"] = df.groupby(group_cols).cumcount() + 1

    # Lag features
    for lag in [1, 2, 3]:
        df[f"ev_total_lag{lag}"] = df.groupby(group_cols)[ev_col].shift(lag)

    # Rolling mean of last 3
    df["ev_total_roll_mean_3"] = (
        df.groupby(group_cols)[ev_col]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # Percentage changes
    lag1 = df["ev_total_lag1"]
    lag2 = df["ev_total_lag2"]
    lag3 = df["ev_total_lag3"]
    df["ev_total_pct_change_1"] = np.where(lag2 != 0, (lag1 - lag2) / lag2, 0)
    df["ev_total_pct_change_3"] = np.where(lag3 != 0, (lag1 - lag3) / lag3, 0)

    # Cumulative EV
    df["cumulative_ev"] = df.groupby(group_cols)[ev_col].cumsum()

    # Growth slope (slope of last 6 cumulative values)
    def compute_slope(series):
        slopes = []
        for i in range(len(series)):
            window = series.iloc[max(0, i - 5):i + 1].values
            if len(window) >= 2:
                slopes.append(np.polyfit(range(len(window)), window, 1)[0])
            else:
                slopes.append(0.0)
        return pd.Series(slopes, index=series.index)

    df["ev_growth_slope"] = df.groupby(group_cols)["cumulative_ev"].transform(compute_slope)

    # ── 6. Fill NaN lags with 0, drop sparse groups ──────────────────────
    lag_cols = ["ev_total_lag1", "ev_total_lag2", "ev_total_lag3"]
    df[lag_cols] = df[lag_cols].fillna(0)

    # Drop rows where we don't have enough history (first 4 rows per group)
    min_rows = 4
    group_sizes = df.groupby(group_cols).cumcount()
    df = df[group_sizes >= min_rows].reset_index(drop=True)

    # Drop counties with fewer than 6 total rows after filtering
    valid_groups = df.groupby(group_cols).size()
    valid_groups = valid_groups[valid_groups >= 6].index
    df = df.set_index(group_cols).loc[valid_groups].reset_index()

    # ── 7. Reorder columns ───────────────────────────────────────────────
    col_order = [
        "Date", "County", "State", "Vehicle Primary Use",
        "Battery Electric Vehicles (BEVs)",
        "Plug-In Hybrid Electric Vehicles (PHEVs)",
        "Electric Vehicle (EV) Total",
        "Non-Electric Vehicle Total",
        "Total Vehicles",
        "Percent Electric Vehicles",
        "year", "month", "numeric_date", "county_encoded",
        "months_since_start",
        "ev_total_lag1", "ev_total_lag2", "ev_total_lag3",
        "ev_total_roll_mean_3",
        "ev_total_pct_change_1", "ev_total_pct_change_3",
        "cumulative_ev", "ev_growth_slope",
    ]
    df = df[col_order]

    # ── 8. Save ──────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✅ Preprocessed data saved to {out_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Counties: {df['County'].nunique()}")
    print(f"   States: {df['State'].nunique()}")
    print(f"   Date range: {df['Date'].min()} → {df['Date'].max()}")


if __name__ == "__main__":
    main()
