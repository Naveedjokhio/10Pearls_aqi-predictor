"""
Training Pipeline for AQI Predictor (Pakistan - Top 6 Cities)
--------------------------------------------------------------------
For each city:
1. Fetches historical (features, target) rows from the Feature Store (BigQuery)
2. Builds 3-day-ahead targets (aqi_day1, aqi_day2, aqi_day3)
3. Trains Ridge Regression, Random Forest, and a small TensorFlow/Keras
   neural network - statistical to deep learning, per the project brief
4. Evaluates with RMSE, MAE, R2 and keeps the best model per horizon
5. Uploads the best model bundle to GCS and registers it in the
   Vertex AI Model Registry (one registry entry per city)

Run:  python training_pipeline/train_model.py
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF's noisy logs

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_pipeline.fetch_features import get_bq_client, get_table_ref, CITIES, FEATURE_COLS

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCS_BUCKET = os.getenv("GCS_BUCKET")
HORIZONS = {"day1": 1, "day2": 2, "day3": 3}
MODEL_DIR = "training_pipeline/artifacts"


class KerasMLPRegressor:
    """Minimal sklearn-style wrapper around a small TensorFlow/Keras MLP,
    so it can be trained/evaluated alongside Ridge and Random Forest using
    the same .fit()/.predict() interface, and safely joblib-pickled.

    Only plain numpy weight arrays are pickled (not the TF graph/session),
    which avoids the serialization issues that come with pickling Keras
    models directly.
    """

    def __init__(self, input_dim, epochs=80, batch_size=8, random_state=42):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._weights = None
        self._mean = None
        self._std = None
        self._model = None  # built lazily, never pickled

    def _build(self):
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)
        model = keras.Sequential([
            keras.layers.Input(shape=(self.input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, X, y):
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")

        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        X_scaled = (X - self._mean) / self._std

        # Scale the target too - AQI values range roughly 0-300+, and
        # training MSE loss directly on that raw scale was destabilizing
        # the network (exploding loss / poor convergence on small datasets).
        self._y_mean = y.mean()
        self._y_std = y.std() if y.std() > 0 else 1.0
        y_scaled = (y - self._y_mean) / self._y_std

        model = self._build()
        model.fit(X_scaled, y_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self._weights = model.get_weights()
        self._model = model
        return self

    def predict(self, X):
        X = np.asarray(X, dtype="float32")
        X_scaled = (X - self._mean) / self._std
        if self._model is None:
            self._model = self._build()
            self._model.set_weights(self._weights)
        preds_scaled = self._model.predict(X_scaled, verbose=0).flatten()
        return preds_scaled * self._y_std + self._y_mean

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None  # never pickle the live TF model object
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None


def load_training_data(city: str) -> pd.DataFrame:
    client = get_bq_client()
    query = f"""
        SELECT event_time, {", ".join(FEATURE_COLS)}
        FROM `{get_table_ref()}`
        WHERE city = @city
        ORDER BY event_time
    """
    from google.cloud import bigquery
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("city", "STRING", city)]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time")

    df["date"] = df["event_time"].dt.date
    daily = df.groupby("date").agg({c: "mean" for c in FEATURE_COLS}).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def build_targets(daily: pd.DataFrame) -> pd.DataFrame:
    for name, shift in HORIZONS.items():
        daily[f"target_{name}"] = daily["aqi"].shift(-shift)
    daily = daily.dropna(subset=[f"target_{n}" for n in HORIZONS]).reset_index(drop=True)
    return daily


def train_and_evaluate(X_train, X_test, y_train, y_test, model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"    {name:20s} RMSE={rmse:7.2f}  MAE={mae:7.2f}  R2={r2:6.3f}")
    return model, {"rmse": rmse, "mae": mae, "r2": r2}


def upload_to_gcs(local_path: str, bucket_name: str, blob_name: str) -> str:
    from google.cloud import storage
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"  Uploaded model to {gcs_uri}")
    return gcs_uri


def register_in_vertex_model_registry(city: str, gcs_dir_uri: str, avg_rmse: float):
    from google.cloud import aiplatform
    aiplatform.init(project=GCP_PROJECT_ID, location=os.getenv("GCP_REGION", "us-central1"))

    model = aiplatform.Model.upload(
        display_name=f"aqi_forecast_{city}",
        artifact_uri=gcs_dir_uri,
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
        ),
        labels={"avg_rmse": str(round(avg_rmse, 2)).replace(".", "_"), "city": city},
    )
    print(f"  Registered model in Vertex AI Model Registry: {model.resource_name}")


def train_city(city: str):
    print(f"\n=== Training models for {city} ===")
    daily = load_training_data(city)

    if len(daily) < 8:
        print(f"  Only {len(daily)} daily rows available for {city}. Need at "
              f"least ~8 days. Skipping (run backfill.py or wait for more data).")
        return

    daily = build_targets(daily)
    X = daily[FEATURE_COLS]

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_models = {}
    best_scores = {}

    for horizon_name in HORIZONS:
        print(f"  --- Horizon: {horizon_name} ---")
        y = daily[f"target_{horizon_name}"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        candidates = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(
                n_estimators=200, max_depth=8, random_state=42
            ),
            "NeuralNet (TensorFlow)": KerasMLPRegressor(input_dim=X_train.shape[1]),
        }

        results = {}
        for name, model in candidates.items():
            trained, metrics = train_and_evaluate(
                X_train, X_test, y_train, y_test, model, name
            )
            results[name] = (trained, metrics)

        best_name = min(results, key=lambda n: results[n][1]["rmse"])
        best_models[horizon_name] = results[best_name][0]
        best_scores[horizon_name] = results[best_name][1]
        print(f"    -> Best for {horizon_name}: {best_name}")

    bundle_path = os.path.join(MODEL_DIR, f"aqi_model_bundle_{city}.joblib")
    joblib.dump(
        {"models": best_models, "feature_cols": FEATURE_COLS, "scores": best_scores},
        bundle_path,
    )
    print(f"  Saved local model bundle to {bundle_path}")

    if not GCS_BUCKET:
        print("  GCS_BUCKET not set in .env - skipping Model Registry upload.")
        return

    avg_rmse = float(np.mean([s["rmse"] for s in best_scores.values()]))
    # Vertex AI's prebuilt sklearn container requires the artifact file to
    # be named exactly "model.joblib" (or "model.pkl") inside its directory.
    blob_name = f"aqi_forecast_{city}/model.joblib"
    upload_to_gcs(bundle_path, GCS_BUCKET, blob_name)

    gcs_dir_uri = f"gs://{GCS_BUCKET}/aqi_forecast_{city}"
    register_in_vertex_model_registry(city, gcs_dir_uri, avg_rmse)


def run():
    for city in CITIES:
        try:
            train_city(city)
        except Exception as e:
            print(f"Training failed for {city}: {e}")

    print("\nTraining pipeline complete for all cities.")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"Training pipeline failed: {e}")
        sys.exit(1)