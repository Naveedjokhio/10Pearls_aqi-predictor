import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_pipeline.fetch_features import get_bq_client, get_table_ref, CITY, FEATURE_COLS

load_dotenv()

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_outputs")


def load_data() -> pd.DataFrame:
    client = get_bq_client()
    query = f"""
        SELECT event_time, {", ".join(FEATURE_COLS)}
        FROM `{get_table_ref()}`
        WHERE city = @city
        ORDER BY event_time
    """
    from google.cloud import bigquery
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("city", "STRING", CITY)]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df.sort_values("event_time")


def plot_aqi_over_time(df: pd.DataFrame):
    plt.figure(figsize=(12, 5))
    plt.plot(df["event_time"], df["aqi"], color="#d35400")
    plt.title("Karachi AQI Over Time")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aqi_over_time.png"))
    plt.close()


def plot_hourly_pattern(df: pd.DataFrame):
    hourly = df.groupby("hour")["aqi"].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=hourly, x="hour", y="aqi", color="#2980b9")
    plt.title("Average AQI by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Average AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aqi_by_hour.png"))
    plt.close()


def plot_weekday_pattern(df: pd.DataFrame):
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday = df.groupby("day_of_week")["aqi"].mean().reset_index()
    weekday["day_name"] = weekday["day_of_week"].apply(lambda x: day_names[int(x)])
    plt.figure(figsize=(10, 5))
    sns.barplot(data=weekday, x="day_name", y="aqi", color="#27ae60")
    plt.title("Average AQI by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aqi_by_weekday.png"))
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    corr_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                 "temperature", "humidity", "pressure", "wind_speed"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"))
    plt.close()


def plot_pollutant_distributions(df: pd.DataFrame):
    pollutants = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flatten(), pollutants):
        sns.histplot(df[col], kde=True, ax=ax, color="#8e44ad")
        ax.set_title(f"{col.upper()} distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pollutant_distributions.png"))
    plt.close()


def print_summary_stats(df: pd.DataFrame):
    print("\n=== Summary Statistics ===")
    print(df[["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
               "temperature", "humidity"]].describe().T)

    print("\n=== AQI change rate stats ===")
    print(df["aqi_change_rate"].describe())

    print(f"\nDate range: {df['event_time'].min()} to {df['event_time'].max()}")
    print(f"Total rows: {len(df)}")


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data from Feature Store (BigQuery)...")
    df = load_data()

    if df.empty:
        print("No data yet. Run feature_pipeline/fetch_features.py and "
              "backfill.py first.")
        return

    print_summary_stats(df)

    print("\nGenerating plots...")
    plot_aqi_over_time(df)
    plot_hourly_pattern(df)
    plot_weekday_pattern(df)
    plot_correlation_heatmap(df)
    plot_pollutant_distributions(df)

    print(f"\nDone. Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    run()
