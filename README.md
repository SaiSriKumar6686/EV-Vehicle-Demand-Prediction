# ⚡ EV Demand Forecaster

**AI-powered Electric Vehicle adoption forecasting across US counties.**

A production-ready Streamlit dashboard that uses a Random Forest model with autoregressive features to forecast EV adoption trends for 269 counties across 51 US states and territories.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange?logo=scikitlearn)

---

## 🚀 Features

- **County-Level Forecasting** — Select any county and generate multi-year forecasts with 80% confidence intervals
- **Multi-County Comparison** — Overlay up to 5 counties for side-by-side analysis
- **Interactive Choropleth Map** — Visualize EV adoption density across US states
- **BEV vs PHEV Breakdown** — Separate analysis of Battery EVs and Plug-in Hybrids
- **Monthly Heatmaps** — Year × month patterns of registration intensity
- **Market Penetration Tracking** — EV share of total vehicle fleet over time
- **County Leaderboards** — Rankings by total EVs, growth rate, and penetration
- **Model Transparency** — Feature importances, evaluation metrics, methodology docs
- **Downloadable Reports** — Export forecasts as CSV or Excel

## 📊 Model Performance

| Metric | Train | Test |
|--------|-------|------|
| **R²** | 0.999 | 0.940 |
| **MAE** | — | 0.064 |
| **RMSE** | — | 0.501 |
| **MAPE** | — | 2.09% |
| **CV R² (5-fold)** | — | 0.982 ± 0.014 |

## 📂 Project Structure

```
EV-Vehicle-Demand-Prediction/
├── app.py                        # Streamlit dashboard (entry point)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container deployment
├── .streamlit/config.toml        # Streamlit theme config
├── src/
│   ├── config.py                 # Paths, colors, constants
│   ├── data_loader.py            # Data loading & caching
│   ├── forecaster.py             # Forecasting engine + CI
│   └── visualizations.py         # Plotly chart builders
├── scripts/
│   ├── preprocess.py             # Raw → processed data pipeline
│   └── train_model.py            # Model training + evaluation
├── data/
│   ├── raw/EV_Data.csv           # Original dataset (20,819 rows)
│   └── processed/                # Feature-engineered data
├── models/
│   ├── forecasting_ev_model.pkl  # Trained model
│   └── model_metrics.json        # Evaluation metrics
└── assets/
    └── ev_car.jpg                # Hero banner
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/EV-Vehicle-Demand-Prediction.git
cd EV-Vehicle-Demand-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### Docker Deployment

```bash
# Build the container
docker build -t ev-forecaster .

# Run
docker run -p 8501:8501 ev-forecaster
```

### Streamlit Cloud

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the entry point

## 🔄 Retraining the Model

To retrain with updated data:

```bash
# Step 1: Place new raw data in data/raw/EV_Data.csv
# Step 2: Run preprocessing
python scripts/preprocess.py

# Step 3: Train and evaluate
python scripts/train_model.py
```

Metrics and model will be saved automatically.

## 📈 Data

- **Source**: US EV registration data
- **Period**: February 2017 — February 2024
- **Scope**: 269 counties, 51 states/territories
- **Vehicle Types**: Passenger & Truck (BEV + PHEV)

## 🧠 How It Works

1. **Feature Engineering**: Temporal lags (1/2/3 months), rolling means, percentage changes, and growth slope
2. **Model**: Random Forest (200 trees, max_depth=15) trained on time-based 80/20 split
3. **Forecasting**: Autoregressive rollforward — each prediction feeds into the next month's features
4. **Confidence Intervals**: 10th–90th percentile of individual tree predictions

## 👤 Author

**SaiSriKumar Parimi**

---

*Built with Streamlit, scikit-learn, and Plotly*
