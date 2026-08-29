import os
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_pipeline.fetch_features import (
    CITIES, OPENWEATHER_API_KEY, compute_overall_aqi,
    get_bq_client, ensure_feature_store_exists, insert_rows
)

load_dotenv()

BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "10"))


def fetch_history(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> list:
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}"
        f"&start={int(start_dt.timestamp())}&end={int(end_dt.timestamp())}"
        f"&appid={OPENWEATHER_API_KEY}"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data.get("list", [])


def build_backfill_rows(records: list, city: str) -> pd.DataFrame:
    rows = []
    for rec in records:
        components = rec.get("components", {})
        obs_time = datetime.fromtimestamp(rec["dt"], tz=timezone.utc)
        pm25 = components.get("pm2_5", 0) or 0
        pm10 = components.get("pm10", 0) or 0

        rows.append({
            "city": city,
            "event_time": obs_time,
            "aqi": compute_overall_aqi(pm25, pm10),
            "pm25": pm25,
            "pm10": pm10,
            "o3": components.get("o3", 0) or 0,
            "no2": components.get("no2", 0) or 0,
            "so2": components.get("so2", 0) or 0,
            "co": components.get("co", 0) or 0,
            "temperature": 0.0,
            "humidity": 0.0,
            "pressure": 0.0,
            "wind_speed": 0.0,
            "hour": obs_time.hour,
            "day": obs_time.day,
            "month": obs_time.month,
            "day_of_week": obs_time.weekday(),
            "is_weekend": int(obs_time.weekday() >= 5),
        })

    df = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    df["aqi_change_rate"] = df["aqi"].pct_change().fillna(0)
    return df


def run():
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=BACKFILL_DAYS)

    print("Connecting to Feature Store (BigQuery / Vertex AI)...")
    client = get_bq_client()
    ensure_feature_store_exists(client)

    for city, (lat, lon) in CITIES.items():
        try:
            print(f"\n--- Backfilling {city} ({start_dt.date()} to {end_dt.date()}) ---")
            records = fetch_history(lat, lon, start_dt, end_dt)

            if not records:
                print(f"No historical records returned for {city}. Skipping.")
                continue

            df = build_backfill_rows(records, city)
            print(f"Built {len(df)} rows for {city}.")

            insert_rows(client, df)
            print(f"Inserted {len(df)} rows for {city}.")
        except Exception as e:
            print(f"Backfill failed for {city}: {e}")

    print("\nBackfill complete for all cities.")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"Backfill failed: {e}")
        sys.exit(1)
