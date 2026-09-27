"""Flask service for real-time flight price predictions."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "flight_price_model.joblib"
app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model is missing. Run flight-price-pred-mlflow.py first.")
    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    required = ["from", "to", "flightType", "agency", "date"]
    missing = [field for field in required if not payload.get(field)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
    try:
        date = datetime.strptime(payload["date"], "%Y-%m-%d")
        features = pd.DataFrame([{
            "from": payload["from"], "to": payload["to"], "flightType": payload["flightType"],
            "agency": payload["agency"], "week_no": date.isocalendar().week, "week_day": date.weekday(), "day": date.day,
        }])
        price = float(load_model().predict(features)[0])
    except ValueError:
        return jsonify({"error": "date must use YYYY-MM-DD format"}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    return jsonify({"predicted_price": round(price, 2), "currency": "BRL"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
