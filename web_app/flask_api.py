"""
Flask REST API for AQI Predictor (Pakistan - Top 6 Cities)
--------------------------------------------------------------
A lightweight JSON API alongside the Streamlit dashboard, so the same
predictions/features are also available programmatically (e.g. for other
apps, mobile clients, or automated checks) - not just through the UI.

Endpoints:
  GET /api/cities                - list of supported cities
  GET /api/aqi/<city>             - current AQI + latest pollutant readings
  GET /api/forecast/<city>        - 3-day AQI forecast (median of all
                                     trained models per horizon)

Run locally:  python web_app/flask_api.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_pipeline.fetch_features import get_bq_client, get_table_ref, CITIES, FEATURE_COLS
from training_pipeline.model_wrapper import KerasMLPRegressor  # noqa: F401 (needed for unpickling)

load_dotenv()

app = Flask(__name__)

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCS_BUCKET = os.getenv("GCS_BUCKET")


def _load_latest_row(city: str):
    client = get_bq_client()
    query = f"""
        SELECT event_time, {", ".join(FEATURE_COLS)}
        FROM `{get_table_ref()}`
        WHERE city = @city
        ORDER BY event_time DESC
        LIMIT 1
    """
    from google.cloud import bigquery
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("city", "STRING", city)]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    if df.empty:
        return None
    return df.iloc[0]


def _load_model_bundle(city: str):
    from google.cloud import storage

    local_path = f"/tmp/aqi_model_bundle_{city}_api.joblib"
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"aqi_forecast_{city}/model.joblib")
    blob.download_to_filename(local_path)
    return joblib.load(local_path)


@app.route("/api/cities", methods=["GET"])
def list_cities():
    return jsonify({"cities": list(CITIES.keys())})


@app.route("/api/aqi/<city>", methods=["GET"])
def current_aqi(city):
    city = city.lower()
    if city not in CITIES:
        return jsonify({"error": f"Unknown city '{city}'. Valid options: {list(CITIES.keys())}"}), 404

    row = _load_latest_row(city)
    if row is None:
        return jsonify({"error": f"No data yet for {city}"}), 404

    return jsonify({
        "city": city,
        "event_time": str(row["event_time"]),
        "aqi": float(row["aqi"]),
        "pm25": float(row["pm25"]),
        "pm10": float(row["pm10"]),
        "temperature": float(row["temperature"]),
        "humidity": float(row["humidity"]),
    })


@app.route("/api/forecast/<city>", methods=["GET"])
def forecast(city):
    city = city.lower()
    if city not in CITIES:
        return jsonify({"error": f"Unknown city '{city}'. Valid options: {list(CITIES.keys())}"}), 404

    row = _load_latest_row(city)
    if row is None:
        return jsonify({"error": f"No data yet for {city}"}), 404

    try:
        bundle = _load_model_bundle(city)
    except Exception as e:
        return jsonify({"error": f"Could not load model for {city}: {e}"}), 500

    X_latest = pd.DataFrame([row[FEATURE_COLS]])
    models = bundle["models"]  # {horizon: {model_name: model}}

    result = {"city": city, "current_aqi": float(row["aqi"])}
    for horizon, horizon_models in models.items():
        preds = [float(m.predict(X_latest)[0]) for m in horizon_models.values()]
        pred = float(np.median(preds))
        pred = max(0.0, min(500.0, pred))
        result[horizon] = round(pred, 1)

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)