<div align="center">

# 🚗⚡ EV Adoption Forecasting using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Model Accuracy](https://img.shields.io/badge/R²%20Score-0.99-brightgreen?style=for-the-badge)](https://github.com/)
[![License](https://img.shields.io/badge/License-Academic-blue?style=for-the-badge)](https://github.com/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)](https://github.com/)

<br/>

> **Predicting county-level Electric Vehicle (EV) adoption trends for the next 36 months using feature-engineered time-series regression and an interactive Streamlit dashboard.**

<br/>

*Internship Project — Edunet Foundation × Shell | B.Tech CSE, Anurag Engineering College*

</div>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Dataset Information](#-dataset-information)
- [System Architecture](#-system-architecture)
- [Technologies Used](#-technologies-used)
- [Feature Engineering](#-feature-engineering-core-strength)
- [Model Details](#-model-details)
- [Training Strategy](#-training-strategy)
- [Forecasting Approach](#-forecasting-approach)
- [Visualization](#-visualization)
- [Streamlit Dashboard](#-deployment--streamlit-dashboard)
- [Challenges & Solutions](#-challenges--solutions)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Model Justification](#-model-justification)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [Author](#-author)

---

## 📌 Project Overview

This project builds an **end-to-end machine learning pipeline** to forecast Electric Vehicle (EV) adoption for the next **36 months** using historical registration data from Washington State (2017–2024).

The system leverages **feature-engineered time-series regression (Random Forest)** and serves predictions through an **interactive Streamlit dashboard**, enabling data-driven decisions for:

| Use Case | Value |
|---|---|
| ⚡ Charging Infrastructure Planning | Predict demand hotspots by county |
| 🔌 Power Grid Optimization | Anticipate load from future EV fleet |
| 🏙️ Urban Development Strategy | Guide sustainable city planning |
| 📊 Policy-Making | Evidence-based EV incentive programs |

---

## 🎯 Objectives

- Forecast EV adoption trends at **county-level granularity**
- Capture **non-linear growth patterns** inherent in EV adoption curves
- Provide **interpretable and actionable insights** for non-technical stakeholders
- Build a **production-ready ML application** deployable for real-world use

---

## 📊 Dataset Information

| Attribute | Details |
|---|---|
| **Source** | Washington State Department of Licensing — EV Registration Dataset |
| **Time Range** | January 2017 – December 2024 |
| **Total Records** | ~20,819 rows |
| **Granularity** | Monthly, per county |

### Key Features

| Feature | Type | Description |
|---|---|---|
| `Date` | Datetime | Registration timestamp (monthly) |
| `County` | Categorical | Geographic region within Washington State |
| `Electric Vehicle (EV) Total` | Numerical (Target) | Total registered EVs per county per month |
| `Vehicle Type` | Categorical | Passenger car / Light truck |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAW DATA                                │
│            Washington State EV Registration CSV                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              DATA CLEANING & PREPROCESSING                      │
│   • Handle missing values (fillna)                              │
│   • Parse date columns                                          │
│   • Aggregate by county & month                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                            │
│   • Lag features (lag1, lag2, lag3)                             │
│   • Rolling statistics (mean, % change)                        │
│   • Growth slope (linear regression on window)                  │
│   • Temporal encoding (months_since_start)                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODEL TRAINING (Random Forest)                     │
│   • 80/20 Train-Test Split                                      │
│   • n_estimators=100                                            │
│   • R² ≈ 0.99 | MAE ≈ 1 EV                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODEL PERSISTENCE (Joblib / .pkl)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         RECURSIVE FORECASTING ENGINE (36 Months)                │
│   Step 1: Seed with last known EV values                        │
│   Step 2: Compute lag + statistical features                    │
│   Step 3: Predict next month                                    │
│   Step 4: Append prediction to history                          │
│   Step 5: Repeat × 36                                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│           STREAMLIT DASHBOARD (Interactive Deployment)          │
│   • County selector                                             │
│   • Multi-county comparison (up to 3)                           │
│   • Historical + forecast visualization                         │
│   • Growth % insights                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Technologies Used

### 🔹 Core Machine Learning Stack

| Library | Version | Purpose |
|---|---|---|
| `Python` | 3.10+ | Primary programming language |
| `Pandas` | 2.0+ | Data manipulation and aggregation |
| `NumPy` | 1.24+ | Numerical computations |
| `Scikit-learn` | 1.3+ | Random Forest model, train-test split, metrics |
| `Joblib` | 1.3+ | Model serialization and persistence |

### 🔹 Visualization & Deployment

| Library | Purpose |
|---|---|
| `Matplotlib` | Static and embedded forecast charts |
| `Streamlit` | Interactive web dashboard deployment |
| `HTML + CSS` | Custom UI styling inside Streamlit |

---

## 🧠 Feature Engineering — Core Strength

> Feature engineering is the single most impactful step in this pipeline — it transforms raw time-series data into a format a Random Forest can learn temporal dependencies from.

### ⏱️ Temporal Feature

| Feature | Formula | Captures |
|---|---|---|
| `months_since_start` | Absolute month index | Long-term secular trend |

### 🔁 Lag Features

| Feature | Definition | Captures |
|---|---|---|
| `lag1` | EV count at `t-1` | Most recent momentum |
| `lag2` | EV count at `t-2` | Short-term memory |
| `lag3` | EV count at `t-3` | Quarterly pattern baseline |

### 📉 Statistical Features

| Feature | Definition | Captures |
|---|---|---|
| `rolling_mean_3` | Mean of `lag1, lag2, lag3` | Smoothed trend direction |
| `pct_change` | `(lag1 - lag2) / lag2` | Short-term growth velocity |
| `growth_slope` | OLS slope over a 3-point window | Directional trend (acceleration/deceleration) |

### Why This Matters

```
Without feature engineering:  Random Forest sees isolated values → no temporal context
With feature engineering:     Random Forest sees trends, velocity, memory → learns growth behavior
```

These engineered features allow the model to understand:
- **Short-term memory** → what happened last month matters
- **Growth direction** → are registrations accelerating or stagnating?
- **Volatility** → are spikes seasonal or structural?

---

## 🤖 Model Details

### Algorithm: Random Forest Regressor

A **Random Forest** is an ensemble of decision trees where each tree is trained on a bootstrapped sample and uses a random feature subset at each split. The final prediction is the **average across all trees**, yielding low variance and high robustness.

```
Random Forest
   ├── Tree 1 (bootstrap sample A, random features)
   ├── Tree 2 (bootstrap sample B, random features)
   ├── ...
   └── Tree N → Average prediction → Final Output
```

### Why Random Forest over alternatives?

| Property | Why It Matters for This Problem |
|---|---|
| Handles non-linearity | EV adoption follows an S-curve, not a straight line |
| No stationarity assumption | Unlike ARIMA, no need for differencing transforms |
| Robust to outliers | Resistant to sparse county data spikes |
| Feature importance | Identifies which lag/stat features drive predictions |
| No feature scaling needed | Trees are invariant to scale |

---

## 🧪 Training Strategy

```
Dataset (80%) → Training Set
Dataset (20%) → Test Set  [time-aware split — no data leakage]
```

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 100 | Sufficient forest diversity without over-computation |
| `random_state` | Fixed | Reproducible results |

### Evaluation Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **R² Score** | ≈ 0.99 | Model explains 99% of variance in EV counts |
| **MAE** | ≈ 1 EV | Mean absolute error of ~1 registered vehicle |

> An R² of 0.99 with an MAE of ~1 EV demonstrates that the feature-engineered lag structure captures the growth signal with extremely high fidelity.

---

## 🔮 Forecasting Approach

### Method: Recursive (Autoregressive) Forecasting

Since future ground-truth EV values are unavailable, the model simulates future states step-by-step using its own prior predictions as input.

```python
# Pseudocode: Recursive Forecasting Loop
history = last_known_values  # Seed from real data

for step in range(36):  # 36-month horizon
    features = build_features(history)   # Lags, rolling stats, slope
    prediction = model.predict(features)
    history.append(prediction)           # Feed prediction back as input
    forecast.append(prediction)
```

### Key Consideration

Each prediction compounds on the prior prediction. Errors accumulate over time, which is why high training accuracy (R² ≈ 0.99) is essential — small per-step errors matter at a 36-month horizon.

---

## 📈 Visualization

### Chart Type: Cumulative EV Growth Graph

The dashboard plots **cumulative registered EVs over time** (historical + forecast), not month-over-month delta.

**Why cumulative?**
- Reflects total fleet size on the road — the key metric for infrastructure planning
- Monotonically increasing curves are more interpretable for policymakers
- Noise in month-to-month deltas is smoothed naturally

---

## 🌐 Deployment — Streamlit Dashboard

### Features

| Feature | Description |
|---|---|
| County Selector | Dropdown to choose any county in the dataset |
| Multi-County Comparison | Overlay up to 3 counties on one chart |
| Historical + Forecast Plot | Seamless continuity from real to predicted data |
| Growth % Insight | Displays forecasted % increase from current to 36-month-out value |

### Model Loading

```python
import joblib
model = joblib.load("forecasting_ev_model.pkl")
# Fast inference — model is deserialized once at app startup
```

---

## 🚧 Challenges & Solutions

| Challenge | Root Cause | Solution Applied |
|---|---|---|
| Missing values in time series | Sparse county-level registrations | `fillna()` with forward fill / zero |
| Non-linear adoption curves | S-curve growth dynamics | Random Forest (non-parametric) |
| Sparse data in small counties | Low EV penetration regions | Rolling mean smoothing as input feature |
| Temporal dependency modeling | Standard ML ignores time order | Lag-based autoregressive features |
| Forecast error compounding | Recursive prediction accumulates noise | High R² (0.99) minimizes per-step error |
| Model serving latency | Re-training on every request | Joblib `.pkl` persistence for fast load |

---

## ⚠️ Limitations

- **Exogenous variables excluded:** The model does not factor in government EV incentives, fuel price fluctuations, charging infrastructure expansion, or macroeconomic conditions — all of which materially impact adoption rates.
- **Historical pattern assumption:** The recursive forecast assumes the statistical distribution of past growth persists into the future. Structural breaks (policy shocks, new EV models) are not captured.
- **Geographic scope:** Trained on Washington State data. Generalization to other states requires retraining on region-specific distributions.
- **Fixed horizon:** Forecast error grows with horizon length. Beyond 24 months, predictions should be treated as directional rather than precise.

---

## 🚀 Future Enhancements

- **Exogenous Variables:** Incorporate fuel prices, federal tax credits, charging station density, and median household income as additional regression features
- **Advanced Models:**
  - `XGBoost` / `LightGBM` for improved gradient-boosted performance on tabular data
  - `LSTM` / `Transformer` for sequence modeling with larger multi-state datasets
  - `Prophet` (Meta) for built-in trend + seasonality decomposition
- **Real-Time Data Pipeline:** Automate monthly data ingestion via Washington State DOL API
- **Cloud Deployment:** Containerize with Docker and deploy on AWS EC2 / GCP Cloud Run / Azure App Service
- **MLOps Integration:** Add model versioning (MLflow), drift detection, and automated retraining triggers

---

## ⚖️ Model Justification

| Model | Why Not Used |
|---|---|
| **Linear Regression** | Assumes linearity; fails to capture the S-curve shape of EV adoption |
| **ARIMA / SARIMA** | Requires stationarity; inflexible with external features; univariate only |
| **LSTM / Deep Learning** | Data volume (~20K rows) too small; high overfitting risk without large corpus |
| **SVR** | Computationally expensive; sensitive to hyperparameter tuning; less interpretable |

### ✅ Final Choice: Random Forest Regressor

> Random Forest provides the optimal balance of **predictive accuracy**, **computational efficiency**, **resistance to overfitting**, and **feature interpretability** for this problem size and structure.

---

## 📌 Key Insights

1. **EV adoption is non-linear and time-dependent** — lag features are essential, not optional
2. **Feature engineering dominates model performance** — the same Random Forest without lag features performs significantly worse
3. **Recursive forecasting is viable** for medium-term horizons (12–24 months) when per-step error is low
4. **County-level granularity reveals adoption heterogeneity** — urban counties grow ~3–5x faster than rural ones
5. **ML + interactive visualization = actionable decision support** — not just a prediction, but a planning tool

---

## 📂 Project Structure

```
ev-adoption-forecasting/
│
├── app.py                        # Streamlit application (UI + inference logic)
├── forecasting_ev_model.pkl      # Serialized Random Forest model (Joblib)
├── preprocessed_ev_data.csv      # Feature-engineered dataset (model input)
├── ev_car.jpg                    # Dashboard UI asset
└── README.md                     # Project documentation (this file)
```

---

## 👨‍💻 Author

<div align="center">

| Field | Details |
|---|---|
| **Name** | Parimi Sai Sri Kumar |
| **Hall Ticket No.** | 23C11A05C6 |
| **College** | Anurag Engineering College, Kodad |
| **Program** | B.Tech Computer Science & Engineering |
| **Year** | 3rd Year (2023–2027 Batch) |
| **University** | Jawaharlal Nehru Technological University Hyderabad (JNTUH) |
| **Internship** | Edunet Foundation × Shell — AI/ML Internship |

</div>

---

## 📜 License

This project is developed for **academic and research purposes** as part of the Edunet Foundation × Shell Internship Program. Redistribution for commercial purposes is not permitted without explicit consent.

---

## ⭐ Final Note

<div align="center">

> *"This project demonstrates how **machine learning + time-aware feature engineering + interactive deployment** can transform raw registration records into actionable intelligence for real-world sustainability challenges like EV infrastructure planning."*

**If you found this project useful, please consider giving it a ⭐ on GitHub.**

</div>
