# 🌍 10Pearls AQI Predictor — Pakistan (Top 6 Cities)

> End-to-end, **100% serverless** Air Quality Index (AQI) forecasting system for Pakistan's six largest cities — Karachi, Lahore, Islamabad, Faisalabad, Rawalpindi, and Peshawar. Predicts AQI for the **next 3 days** per city using machine learning (Ridge, Random Forest, TensorFlow Neural Network), with automated CI/CD pipelines, SHAP explainability, and a live Streamlit dashboard.

---

## 🚀 Live Deployments

| Service | URL |
|---|---|
| **Dashboard (Streamlit)** | [aqi-predictor-dashboard](https://aqi-predictor-dashboard-444682761540.asia-south1.run.app) |
| **REST API (Flask)** | [aqi-predictor-api](https://aqi-predictor-api-444682761540.asia-south1.run.app) |

Both services are deployed on **Google Cloud Run** — publicly accessible 24/7, fully serverless (scales to zero when idle).

---

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                            │
│                                                                          │
│  OpenWeather Air Pollution API ──► fetch_features.py (hourly, all cities)│
│  OpenWeather History API ────────► backfill.py (one-time seed, ~10 days) │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     FEATURE STORE (BigQuery / Vertex AI)                 │
│                                                                          │
│  Dataset: aqi_predictor                                                  │
│  Table:   aqi_features_pakistan                                           │
│  Fields:  city, event_time, aqi, pm25, pm10, o3, no2, so2, co,          │
│           temperature, humidity, pressure, wind_speed,                    │
│           hour, day, month, day_of_week, is_weekend, aqi_change_rate     │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE (Daily)                         │
│                                                                          │
│  Per city × per horizon (Day 1, 2, 3):                                  │
│    • Ridge Regression                                                    │
│    • Random Forest (200 trees, max_depth=8)                              │
│    • TensorFlow MLP (32→16→1 neurons, Adam)                             │
│  Metrics: RMSE, MAE, R²  │  Best model selected per horizon             │
│                                                                          │
│  Model Bundle ──► GCS Bucket ──► Vertex AI Model Registry               │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        SERVING LAYER (Cloud Run)                         │
│                                                                          │
│  Streamlit Dashboard:                                                    │
│    • City selector (6 cities)      • 3-day AQI forecast chart           │
│    • All-cities overview strip     • Hazardous AQI alerts               │
│    • SHAP feature importance       • Model performance metrics          │
│    • Raw sensor data table + CSV   • Confidence indicators              │
│                                                                          │
│  Flask REST API:                                                         │
│    • GET /api/cities               • GET /api/aqi/<city>                │
│    • GET /api/forecast/<city>                                           │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      CI/CD (GitHub Actions)                              │
│                                                                          │
│  Hourly:  Feature pipeline   (cron: 0 * * * *)                          │
│  Daily:   Training pipeline  (cron: 0 3 * * *)                          │
│  Auth:    Workload Identity Federation (keyless, OIDC)                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## ✅ Requirements Checklist

| Requirement | Status | Details |
|---|:---:|---|
| Feature Pipeline (fetch / compute / store) | ✅ | `fetch_features.py` — hourly via GitHub Actions |
| Historical Backfill | ✅ | `backfill.py` — real hourly data (~10 days) via OpenWeather history API |
| Training Pipeline (RF, Ridge, RMSE/MAE/R²) | ✅ | `train_model.py` — trains 3 models × 3 horizons × 6 cities |
| TensorFlow / Deep Learning Model | ✅ | `KerasMLPRegressor` (32→16→1 MLP) in `model_wrapper.py` |
| Feature Store + Model Registry | ✅ | BigQuery Feature Store + Vertex AI Model Registry |
| Automated CI/CD (hourly + daily) | ✅ | GitHub Actions with Workload Identity Federation |
| Web Dashboard (Streamlit) | ✅ | Deployed on Cloud Run |
| Flask REST API | ✅ | Deployed as separate Cloud Run service |
| EDA (Exploratory Data Analysis) | ✅ | `notebooks/eda.py` — time series, hourly/weekly patterns, correlations |
| SHAP Explainability | ✅ | Tree/Linear/Kernel SHAP on the dashboard |
| Hazardous AQI Alerts | ✅ | Dashboard alerts when any forecast ≥ 151 |
| 100% Serverless | ✅ | Cloud Run + BigQuery + GCS + Vertex AI |
| Multi-city Support (bonus) | ✅ | 6 cities: Karachi, Lahore, Islamabad, Faisalabad, Rawalpindi, Peshawar |

---

## 📁 Project Structure

```
aqi-predictor-pearls/
├── feature_pipeline/
│   ├── fetch_features.py       # Hourly data ingestion — fetches air pollution + weather
│   │                           #   data from OpenWeather API, computes EPA AQI, writes to
│   │                           #   BigQuery Feature Store for all 6 cities
│   └── backfill.py             # One-time historical backfill — seeds ~10 days of real
│                               #   hourly history using OpenWeather's history endpoint
│
├── training_pipeline/
│   ├── train_model.py          # Trains Ridge, RandomForest, and TensorFlow MLP per city
│   │                           #   per horizon (Day 1/2/3). Saves best model bundles to
│   │                           #   GCS and registers in Vertex AI Model Registry
│   ├── model_wrapper.py        # KerasMLPRegressor — scikit‑learn‑compatible TensorFlow
│   │                           #   wrapper with internal normalization and serialization
│   └── artifacts/              # Local model bundle cache (*.joblib, gitignored)
│
├── web_app/
│   ├── app.py                  # Streamlit dashboard — city selector, forecast chart,
│   │                           #   SHAP explanations, alerts, raw data table, CSV export
│   └── flask_api.py            # Flask REST API — /api/cities, /api/aqi/<city>,
│                               #   /api/forecast/<city>
│
├── notebooks/
│   └── eda.py                  # Exploratory Data Analysis — AQI time series, hourly &
│                               #   weekly patterns, correlation heatmap, distributions
│
├── .github/workflows/
│   ├── feature_pipeline.yml    # Hourly GitHub Actions workflow (cron: every hour)
│   └── training_pipeline.yml   # Daily GitHub Actions workflow (cron: 3 AM UTC)
│
├── .streamlit/
│   └── config.toml             # Dark theme config (deep orange accent, dark background)
│
├── Dockerfile                  # Streamlit dashboard container for Cloud Run
├── requirements.txt            # All Python dependencies (pinned versions)
├── env.yaml                    # Cloud Run environment variable configuration
├── .env.example                # Template for local environment variables
└── .gitignore                  # Ignores .env, keys, venv, model artifacts, caches
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **Data Source** | OpenWeather Air Pollution API + Weather API |
| **Feature Store** | Google BigQuery (backing Vertex AI) |
| **Model Registry** | Vertex AI Model Registry (via GCS staging) |
| **ML Models** | scikit-learn (Ridge, Random Forest), TensorFlow/Keras (MLP) |
| **Explainability** | SHAP (Tree, Linear, Kernel explainers) |
| **Dashboard** | Streamlit + Plotly (interactive charts) |
| **REST API** | Flask |
| **CI/CD** | GitHub Actions (Workload Identity Federation — keyless) |
| **Deployment** | Google Cloud Run (fully serverless, auto-scaling) |
| **Containerization** | Docker |

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- A Google Cloud Platform project with billing enabled
- An [OpenWeather API key](https://openweathermap.org/api) (free tier works)

### 1. Clone & Set Up Environment

```bash
git clone <repo-url>
cd aqi-predictor-pearls

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual values:
#   OPENWEATHER_API_KEY=<your-key>
#   GCP_PROJECT_ID=<your-project>
#   GCS_BUCKET=<your-bucket>
```

### 3. Authenticate with GCP

```bash
gcloud auth application-default login
```

### 4. Enable Required GCP APIs

```bash
gcloud services enable \
  bigquery.googleapis.com \
  storage.googleapis.com \
  aiplatform.googleapis.com \
  iamcredentials.googleapis.com \
  run.googleapis.com
```

### 5. Run Pipelines

```bash
# Test the live feature pipeline (fetches current data for all 6 cities)
python feature_pipeline/fetch_features.py

# Backfill ~10 days of real historical data
python feature_pipeline/backfill.py

# Exploratory Data Analysis
python notebooks/eda.py

# Train models (per city, per horizon)
python training_pipeline/train_model.py

# Launch the dashboard locally
streamlit run web_app/app.py

# Launch the Flask API locally
python web_app/flask_api.py
```

---

## 🌐 Flask REST API Reference

Base URL (live): `https://aqi-predictor-api-444682761540.asia-south1.run.app`

### `GET /api/cities`

Returns the list of supported cities.

```json
{
  "cities": ["karachi", "lahore", "islamabad", "faisalabad", "rawalpindi", "peshawar"]
}
```

### `GET /api/aqi/<city>`

Returns the latest AQI reading for a city.

```json
{
  "city": "karachi",
  "event_time": "2025-06-15 08:00:00+00:00",
  "aqi": 127.0,
  "pm25": 42.3,
  "pm10": 68.1,
  "temperature": 34.0,
  "humidity": 62.0
}
```

### `GET /api/forecast/<city>`

Returns the 3-day AQI forecast for a city.

```json
{
  "city": "karachi",
  "current_aqi": 127.0,
  "day1": 134.2,
  "day2": 119.8,
  "day3": 112.5
}
```

---

## 🐳 Deploying to Cloud Run

### Dashboard (Streamlit)

```bash
gcloud run deploy aqi-predictor-dashboard \
  --source . \
  --region asia-south1 \
  --service-account aqi-predictor-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_REGION=asia-south1,BQ_DATASET=aqi_predictor,BQ_TABLE=aqi_features_pakistan,GCS_BUCKET=YOUR_BUCKET \
  --allow-unauthenticated
```

### API (Flask)

Deploy `flask_api.py` as a separate Cloud Run service with the same environment variables, changing the entrypoint to use Flask instead of Streamlit.

---

## 🔐 Authentication (No Service Account Keys)

This project enforces the `iam.disableServiceAccountKeyCreation` organization policy. All authentication is keyless:

| Environment | Method |
|---|---|
| **Local Development** | `gcloud auth application-default login` (uses your Google login) |
| **GitHub Actions** | Workload Identity Federation (GitHub OIDC → short-lived GCP credentials) |
| **Cloud Run** | Attached service account (metadata server provides credentials automatically) |

### GitHub Repository Secrets Required

| Secret | Description |
|---|---|
| `OPENWEATHER_API_KEY` | OpenWeather API key |
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCS_BUCKET` | GCS bucket for model staging |
| `WIF_PROVIDER` | Workload Identity Federation provider resource name |
| `WIF_SERVICE_ACCOUNT` | Service account email for WIF |

---

## 🧠 How It Works

### Feature Pipeline (`fetch_features.py`)

1. Calls OpenWeather **Air Pollution API** for each city (PM2.5, PM10, O₃, NO₂, SO₂, CO)
2. Calls OpenWeather **Weather API** for meteorological context (temperature, humidity, pressure, wind speed)
3. Computes **EPA-standard AQI** from PM2.5 and PM10 using breakpoint-based interpolation
4. Engineers temporal features: hour, day, month, day of week, weekend flag
5. Computes `aqi_change_rate` relative to the last reading
6. Writes the feature row to **BigQuery Feature Store**

Runs **every hour** via GitHub Actions.

### Training Pipeline (`train_model.py`)

1. Loads daily-aggregated features from BigQuery per city
2. Builds shifted targets: Day 1, Day 2, Day 3 ahead AQI
3. Trains three candidate models per horizon:
   - **Ridge Regression** (α=1.0) — linear baseline
   - **Random Forest** (200 trees, max_depth=8) — ensemble method
   - **TensorFlow MLP** (32→16→1, ReLU, Adam, MSE loss) — deep learning
4. Evaluates on 80/20 holdout split using **RMSE**, **MAE**, **R²**
5. Saves all trained models as a **joblib bundle** per city
6. Uploads to **GCS** and registers in **Vertex AI Model Registry**

Runs **daily at 3 AM UTC** via GitHub Actions.

### Streamlit Dashboard (`app.py`)

- **City Selector**: switch between all 6 cities
- **All-Cities Overview**: horizontal strip showing live AQI + color-coded badges for every city
- **City Ranking**: shows the selected city's rank among all tracked cities
- **Current Conditions**: AQI badge, PM2.5, temperature, humidity/wind, last updated timestamp
- **3-Day Forecast Chart**: interactive Plotly chart with color-coded markers per AQI band
- **Confidence Indicators**: per-horizon confidence based on median R² of candidate models
- **Hazardous Alerts**: warning cards for any forecasted AQI ≥ 151
- **SHAP Explanations**: feature importance bar chart for the Day 1 prediction
- **Model Performance**: expandable table showing RMSE/MAE/R² for all models × horizons
- **Raw Data Table**: last 20 sensor readings with CSV download

---

## 📊 AQI Classification (EPA Standard)

| AQI Range | Category | Color |
|---|---|---|
| 0–50 | Good | 🟢 |
| 51–100 | Moderate | 🟡 |
| 101–150 | Unhealthy for Sensitive Groups | 🟠 |
| 151–200 | Unhealthy | 🔴 |
| 201–300 | Very Unhealthy | 🟣 |
| 301–500 | Hazardous | 🟤 |

---

## 📝 Important Notes

### Data Source Decision

The project instructions allowed **AQICN or OpenWeather**. AQICN was tried first, but it has **no active ground station for Karachi or the wider Sindh region** (confirmed via AQICN's `map/bounds` API returning zero stations — the only historical Karachi station, US Consulate, stopped reporting in March 2025). The project therefore uses **OpenWeather's Air Pollution API**, which is satellite/model-based and has full coverage across all 6 cities.

### Feature Store Decision (Vertex AI vs Hopsworks)

The project instructions allowed **Hopsworks or Vertex AI**. Hopsworks was initially attempted but their platform was experiencing a **server-side outage** (confirmed via their status page: *"We are currently experiencing issues with data ingestion and reads from external environments"*). Since Vertex AI (BigQuery Feature Store + Vertex AI Model Registry) was an explicitly allowed alternative, the project uses the **full GCP-native stack** — a legitimate, documented engineering decision.

### Historical Data Caveat

Weather fields (temperature, humidity, pressure, wind) are **not available** on OpenWeather's free historical tier, so backfilled rows have these set to `0.0`. Every row collected going forward via the hourly pipeline has full weather data. This does not affect the AQI target itself (computed from PM2.5/PM10, both fully available historically).

---

## 📄 License

This project was developed as part of the **10Pearls University** program.
