"""Airflow DAG that retrains the flight-price model on a weekly schedule.

Place this file in Airflow's dags directory. Airflow must run in an environment
where this repository is mounted at the path in PROJECT_ROOT.
"""
from datetime import datetime
from pathlib import Path
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCRIPT = PROJECT_ROOT / "Productionisation_Travel_ML_System" / "Productionisation_ML_Systems" / "flight-price-pred-mlflow.py"


def train_model():
    subprocess.run([sys.executable, str(TRAINING_SCRIPT)], cwd=PROJECT_ROOT, check=True)


with DAG(
    dag_id="flight_price_retraining",
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["travel", "mlops"],
) as dag:
    PythonOperator(task_id="train_and_log_model", python_callable=train_model)
