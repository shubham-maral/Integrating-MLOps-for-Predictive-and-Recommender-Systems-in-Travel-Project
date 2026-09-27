"""Train a reproducible gender-classification baseline from users.csv.

This is an educational classification exercise. Do not use its predictions for
eligibility, pricing, employment, or other consequential decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent / "model"


def train() -> dict[str, float]:
    users = pd.read_csv(ROOT / "users.csv")
    users = users[users["gender"].isin(["male", "female"])].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        users[["name", "company", "age"]], users["gender"], test_size=0.2,
        random_state=42, stratify=users["gender"],
    )
    features = ColumnTransformer([
        ("name", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2), "name"),
        ("company", OneHotEncoder(handle_unknown="ignore"), ["company"]),
        ("age", StandardScaler(), ["age"]),
    ])
    model = Pipeline([("features", features), ("classifier", LogisticRegression(max_iter=1000))])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {"accuracy": float(accuracy_score(y_test, predictions))}
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "gender_classifier.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(classification_report(y_test, predictions))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
