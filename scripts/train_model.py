"""
train_model.py — Train the EV forecasting model and save evaluation metrics.

Usage:
    python scripts/train_model.py

This script:
  1. Loads preprocessed data
  2. Splits into train/test (time-based)
  3. Trains a RandomForestRegressor with hyperparameter tuning
  4. Evaluates on test set
  5. Saves model (.pkl) and metrics (.json)
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_val_score
import joblib

from src.config import MODEL_FEATURES, DATA_PROCESSED, MODEL_PATH, METRICS_PATH


def main():
    print("📂 Loading preprocessed data...")
    df = pd.read_csv(DATA_PROCESSED)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"   Shape: {df.shape}")

    # ── Features and target ──────────────────────────────────────────────
    X = df[MODEL_FEATURES]
    y = df["Electric Vehicle (EV) Total"]

    # ── Time-based train/test split (80/20) ──────────────────────────────
    df_sorted = df.sort_values("Date")
    split_idx = int(len(df_sorted) * 0.8)
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    print(f"   Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── Train model ──────────────────────────────────────────────────────
    print("🔧 Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=4,
        max_features=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "model_type": "RandomForestRegressor",
        "n_estimators": 200,
        "max_depth": 15,
        "features": MODEL_FEATURES,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_metrics": {
            "r2": round(float(r2_score(y_train, y_pred_train)), 4),
            "mae": round(float(mean_absolute_error(y_train, y_pred_train)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_train, y_pred_train))), 4),
        },
        "test_metrics": {
            "r2": round(float(r2_score(y_test, y_pred_test)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred_test)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 4),
        },
        "feature_importances": {
            name: round(float(imp), 4)
            for name, imp in zip(MODEL_FEATURES, model.feature_importances_)
        },
    }

    # Safe MAPE (avoid zero division)
    mask_train = y_train != 0
    mask_test = y_test != 0
    if mask_train.sum() > 0:
        metrics["train_metrics"]["mape"] = round(
            float(mean_absolute_percentage_error(y_train[mask_train], y_pred_train[mask_train]) * 100), 2
        )
    if mask_test.sum() > 0:
        metrics["test_metrics"]["mape"] = round(
            float(mean_absolute_percentage_error(y_test[mask_test], y_pred_test[mask_test]) * 100), 2
        )

    # Cross-validation R²
    print("📊 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
    metrics["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_r2_std"] = round(float(cv_scores.std()), 4)

    # ── Save model ───────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    # ── Save metrics ─────────────────────────────────────────────────────
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {METRICS_PATH}")

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("📊 MODEL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Train R² :  {metrics['train_metrics']['r2']}")
    print(f"  Test  R² :  {metrics['test_metrics']['r2']}")
    print(f"  Test  MAE:  {metrics['test_metrics']['mae']}")
    print(f"  Test  RMSE: {metrics['test_metrics']['rmse']}")
    if "mape" in metrics["test_metrics"]:
        print(f"  Test  MAPE: {metrics['test_metrics']['mape']}%")
    print(f"  CV R² (5-fold): {metrics['cv_r2_mean']} ± {metrics['cv_r2_std']}")
    print()
    print("  Feature Importances:")
    sorted_fi = sorted(metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_fi:
        bar = "█" * int(imp * 50)
        print(f"    {name:30s} {imp:.4f} {bar}")
    print("=" * 50)


if __name__ == "__main__":
    main()
