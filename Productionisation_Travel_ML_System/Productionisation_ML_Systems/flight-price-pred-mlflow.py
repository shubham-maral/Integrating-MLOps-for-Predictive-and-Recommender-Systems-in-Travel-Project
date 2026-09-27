"""Train and version the flight-price regression model.

Run from the repository root:
    py -3.11 Productionisation_Travel_ML_System/Productionisation_ML_Systems/flight-price-pred-mlflow.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "flights.csv"
MODEL_DIR = Path(__file__).resolve().parent / "model"


def make_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create features available at booking time and the flight-price target."""
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    frame["week_no"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["week_day"] = frame["date"].dt.weekday
    frame["day"] = frame["date"].dt.day
    features = frame[["from", "to", "flightType", "agency", "week_no", "week_day", "day"]]
    return features, frame["price"]


def train() -> dict[str, float]:
    data = pd.read_csv(DATA_PATH)
    features, target = make_features(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    categorical = ["from", "to", "flightType", "agency"]
    numeric = ["week_no", "week_day", "day"]
    preprocessing = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical), ("numeric", "passthrough", numeric)]
    )
    model = Pipeline(
        [("preprocessing", preprocessing), ("regressor", RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1))]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "flight_price_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_experiment("flight-price-prediction")
        with mlflow.start_run():
            mlflow.log_params({"model": "RandomForestRegressor", "test_size": 0.2, "random_state": 42})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "flight_price_model")
    except ImportError:
        print("MLflow is not installed; saved local artifacts only.")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
