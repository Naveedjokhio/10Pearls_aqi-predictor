# Pearls AQI Predictor — Pakistan (Top 6 Cities)

End-to-end, serverless AQI (Air Quality Index) forecasting system covering
Karachi, Lahore, Islamabad, Faisalabad, Rawalpindi, and Peshawar. Predicts
AQI for the next 3 days per city using a Feature Store + Model Registry
(Vertex AI, backed by BigQuery), automated pipelines (GitHub Actions), and
a Streamlit dashboard with a city selector, deployed serverlessly on
Google Cloud Run.

## Architecture

```
OpenWeather API → Feature Pipeline (hourly, all cities) → BigQuery Feature Store
                                              ↓
                    Training Pipeline (daily, per city) → GCS + Vertex AI Model Registry
                                              ↓
              Streamlit Web App (city selector) → 3-day AQI Forecast → Cloud Run
```

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your real values:
   - `OPENWEATHER_API_KEY`: from https://openweathermap.org/api
   - `GCP_PROJECT_ID`: your GCP project
   - `GCS_BUCKET`: a GCS bucket to stage trained models

3. Authenticate locally (no service account key needed):
   ```
   gcloud auth application-default login
   ```

4. Enable required GCP APIs (one-time):
   ```
   gcloud services enable bigquery.googleapis.com storage.googleapis.com aiplatform.googleapis.com iamcredentials.googleapis.com run.googleapis.com
   ```

5. Run the feature pipeline once to test (fetches all 6 cities):
   ```
   python feature_pipeline/fetch_features.py
   ```

## Project Structure

```
feature_pipeline/    - fetches raw data per city, computes features, writes to BigQuery Feature Store
training_pipeline/   - trains one model bundle per city, registers in Vertex AI Model Registry
web_app/              - Streamlit dashboard with city selector, forecast, SHAP, alerts
.github/workflows/    - CI/CD: hourly feature runs, daily training runs (via Workload Identity Federation)
notebooks/            - EDA script
Dockerfile             - for deploying the dashboard to Cloud Run
```

## Usage order

1. `python feature_pipeline/fetch_features.py` — test the live fetch for all cities
2. `python feature_pipeline/backfill.py` — seed historical data (real hourly data, ~10 days) for all cities
3. Set up Workload Identity Federation (see below) so GitHub Actions can
   authenticate to GCP without a service account key
4. Push to GitHub, add repo secrets: `OPENWEATHER_API_KEY`, `GCP_PROJECT_ID`, `GCS_BUCKET`
5. Let the hourly feature pipeline run for a few days to build real history
6. `python notebooks/eda.py` — explore trends
7. `python training_pipeline/train_model.py` — train and register models (one per city)
8. `streamlit run web_app/app.py` — view the dashboard locally
9. Deploy to Cloud Run for a fully serverless, publicly accessible dashboard (see below)

## Authentication notes (no service account keys)

This GCP project enforces the `iam.disableServiceAccountKeyCreation`
organization policy, so service account **key files cannot be created**.
The project works around this everywhere:
- **Local development**: `gcloud auth application-default login` (uses your own Google login)
- **GitHub Actions**: Workload Identity Federation (keyless, GitHub OIDC token exchanged for short-lived GCP credentials)
- **Cloud Run**: an attached service account (Cloud Run's metadata server provides credentials automatically, no key needed)

## Deploying the dashboard to Cloud Run

```
gcloud run deploy aqi-predictor-dashboard \
  --source . \
  --region asia-south1 \
  --service-account aqi-predictor-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_REGION=asia-south1,BQ_DATASET=aqi_predictor,BQ_TABLE=aqi_features_pakistan,GCS_BUCKET=YOUR_BUCKET \
  --allow-unauthenticated
```

## Important note on the data source

The project instructions allow **AQICN or OpenWeather**. AQICN was tried
first, but it has **no active ground station for Karachi or the wider Sindh
region** (confirmed via AQICN's own `map/bounds` API returning zero
stations, and the only historical Karachi station — US Consulate — stopped
reporting in March 2025). The project therefore uses **OpenWeather's Air
Pollution API**, which is satellite/model-based and has full coverage
across all 6 cities.

OpenWeather also provides a free **historical** endpoint (`air_pollution/history`,
data back to Nov 2020), so `backfill.py` seeds the Feature Store with real
hourly history rather than an approximation.

## Important note on historical data

Weather fields (temperature, humidity, pressure, wind) are not available on
OpenWeather's free historical tier, so backfilled rows have these set to 0.
Every row collected going forward via the hourly `fetch_features.py` run has
full weather data. This does not affect the AQI target itself (computed
from PM2.5/PM10, both fully available historically).

## Status
- [x] Project scaffolding
- [x] Multi-city support (Karachi, Lahore, Islamabad, Faisalabad, Rawalpindi, Peshawar)
- [x] Feature pipeline (fetch + compute + store in BigQuery Feature Store)
- [x] Historical backfill (real hourly data via OpenWeather history endpoint)
- [x] Training pipeline (Ridge + Random Forest, per-city, per-horizon best model)
- [x] Model Registry (GCS staging + Vertex AI registration, per city)
- [x] CI/CD automation (GitHub Actions: hourly + daily, via Workload Identity Federation)
- [x] Streamlit dashboard with city selector (forecast chart + alerts)
- [x] SHAP explainability
- [x] EDA script
- [ ] Cloud Run deployment (dashboard live + public)
- [ ] Final report
