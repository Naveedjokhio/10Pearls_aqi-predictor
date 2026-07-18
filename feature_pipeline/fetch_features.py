"""
Feature Pipeline for AQI Predictor (Pakistan - Top 6 Cities)
----------------------------------------------------------------
1. Fetches raw weather + pollution data from OpenWeather for each city
   (AQICN has no active ground station for Karachi/Sindh - confirmed via
   their map/bounds API returning zero stations - so OpenWeather is used,
   the other API explicitly allowed by the project instructions)
2. Computes time-based and derived features, and a US EPA AQI from
   pollutant concentrations
3. Stores the features in the Feature Store (BigQuery, backing Vertex AI),
   one row per city per run

Run manually:  python feature_pipeline/fetch_features.py
Run backfill:  python feature_pipeline/backfill.py
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import bigquery

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Top 6 Pakistani cities by population / relevance, with coordinates
CITIES = {
    "karachi": (24.8607, 67.0011),
    "lahore": (31.5497, 74.3436),
    "islamabad": (33.6844, 73.0479),
    "faisalabad": (31.4504, 73.1350),
    "rawalpindi": (33.5651, 73.0169),
    "peshawar": (34.0151, 71.5249),
}

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET", "aqi_predictor")
BQ_TABLE = os.getenv("BQ_TABLE", "aqi_features_pakistan")

FEATURE_COLS = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "hour", "day", "month", "day_of_week", "is_weekend",
    "aqi_change_rate", "aqi",
]

TABLE_SCHEMA = [
    bigquery.SchemaField("city", "STRING"),
    bigquery.SchemaField("event_time", "TIMESTAMP"),
    bigquery.SchemaField("aqi", "FLOAT"),
    bigquery.SchemaField("pm25", "FLOAT"),
    bigquery.SchemaField("pm10", "FLOAT"),
    bigquery.SchemaField("o3", "FLOAT"),
    bigquery.SchemaField("no2", "FLOAT"),
    bigquery.SchemaField("so2", "FLOAT"),
    bigquery.SchemaField("co", "FLOAT"),
    bigquery.SchemaField("temperature", "FLOAT"),
    bigquery.SchemaField("humidity", "FLOAT"),
    bigquery.SchemaField("pressure", "FLOAT"),
    bigquery.SchemaField("wind_speed", "FLOAT"),
    bigquery.SchemaField("hour", "INTEGER"),
    bigquery.SchemaField("day", "INTEGER"),
    bigquery.SchemaField("month", "INTEGER"),
    bigquery.SchemaField("day_of_week", "INTEGER"),
    bigquery.SchemaField("is_weekend", "INTEGER"),
    bigquery.SchemaField("aqi_change_rate", "FLOAT"),
]

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300),
    (425, 504, 301, 400), (505, 604, 401, 500),
]


def _epa_aqi(conc: float, breakpoints) -> float:
    if conc is None:
        return 0.0
    for lo, hi, aqi_lo, aqi_hi in breakpoints:
        if lo <= conc <= hi:
            return ((aqi_hi - aqi_lo) / (hi - lo)) * (conc - lo) + aqi_lo
    lo, hi, aqi_lo, aqi_hi = breakpoints[-1]
    if conc > hi:
        return float(aqi_hi)
    return 0.0


def compute_overall_aqi(pm25: float, pm10: float) -> float:
    return round(max(_epa_aqi(pm25, PM25_BREAKPOINTS), _epa_aqi(pm10, PM10_BREAKPOINTS)), 1)


def get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=GCP_PROJECT_ID)


def get_table_ref() -> str:
    return f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"


def ensure_feature_store_exists(client: bigquery.Client) -> None:
    dataset_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET}"
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        print(f"Creating Feature Store dataset {dataset_ref}...")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset, exists_ok=True)

    table_ref = get_table_ref()
    try:
        client.get_table(table_ref)
    except Exception:
        print(f"Creating Feature Store table {table_ref}...")
        table = bigquery.Table(table_ref, schema=TABLE_SCHEMA)
        client.create_table(table, exists_ok=True)


def fetch_raw_data(lat: float, lon: float) -> dict:
    if not OPENWEATHER_API_KEY:
        raise ValueError("OPENWEATHER_API_KEY not set. Check your .env file.")

    pollution_url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    )
    weather_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    )

    pollution_resp = requests.get(pollution_url, timeout=15)
    pollution_resp.raise_for_status()
    pollution_data = pollution_resp.json()

    weather_resp = requests.get(weather_url, timeout=15)
    weather_resp.raise_for_status()
    weather_data = weather_resp.json()

    if "list" not in pollution_data or not pollution_data["list"]:
        raise RuntimeError(f"OpenWeather Air Pollution API error: {pollution_data}")

    return {"pollution": pollution_data["list"][0], "weather": weather_data}


def compute_features(raw: dict, city: str) -> pd.DataFrame:
    components = raw["pollution"].get("components", {})
    weather = raw["weather"]

    obs_ts = raw["pollution"].get("dt")
    obs_time = (
        datetime.fromtimestamp(obs_ts, tz=timezone.utc) if obs_ts
        else datetime.now(timezone.utc)
    )

    pm25 = components.get("pm2_5", 0) or 0
    pm10 = components.get("pm10", 0) or 0
    overall_aqi = compute_overall_aqi(pm25, pm10)

    row = {
        "city": city,
        "event_time": obs_time,
        "aqi": overall_aqi,
        "pm25": pm25,
        "pm10": pm10,
        "o3": components.get("o3", 0) or 0,
        "no2": components.get("no2", 0) or 0,
        "so2": components.get("so2", 0) or 0,
        "co": components.get("co", 0) or 0,
        "temperature": weather.get("main", {}).get("temp", 0) or 0,
        "humidity": weather.get("main", {}).get("humidity", 0) or 0,
        "pressure": weather.get("main", {}).get("pressure", 0) or 0,
        "wind_speed": weather.get("wind", {}).get("speed", 0) or 0,
        "hour": obs_time.hour,
        "day": obs_time.day,
        "month": obs_time.month,
        "day_of_week": obs_time.weekday(),
        "is_weekend": int(obs_time.weekday() >= 5),
    }

    return pd.DataFrame([row])


def get_last_aqi_value(client: bigquery.Client, city: str):
    query = f"""
        SELECT aqi FROM `{get_table_ref()}`
        WHERE city = @city
        ORDER BY event_time DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("city", "STRING", city)]
    )
    try:
        results = list(client.query(query, job_config=job_config).result())
        return float(results[0]["aqi"]) if results else None
    except Exception as e:
        print(f"Could not read previous AQI for {city} (probably first run): {e}")
        return None


def add_aqi_change_rate(df: pd.DataFrame, previous_aqi) -> pd.DataFrame:
    if previous_aqi is not None and previous_aqi != 0:
        df["aqi_change_rate"] = (df["aqi"] - previous_aqi) / previous_aqi
    else:
        df["aqi_change_rate"] = 0.0
    return df


def insert_rows(client: bigquery.Client, df: pd.DataFrame) -> None:
    table_ref = get_table_ref()
    job_config = bigquery.LoadJobConfig(
        schema=TABLE_SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()


def run():
    print("Connecting to Feature Store (BigQuery / Vertex AI)...")
    client = get_bq_client()
    ensure_feature_store_exists(client)

    for city, (lat, lon) in CITIES.items():
        try:
            print(f"\n--- {city} (lat={lat}, lon={lon}) ---")
            raw = fetch_raw_data(lat, lon)
            df = compute_features(raw, city)

            previous_aqi = get_last_aqi_value(client, city)
            df = add_aqi_change_rate(df, previous_aqi)

            print(df[["city", "event_time", "aqi", "pm25", "pm10"]].to_string(index=False))

            insert_rows(client, df)
            print(f"Row inserted for {city}.")
        except Exception as e:
            print(f"Failed to fetch/insert data for {city}: {e}")

    print("\nDone. Feature pipeline run complete for all cities.")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"Feature pipeline failed: {e}")
        sys.exit(1)
