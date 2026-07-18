"""
Streamlit Dashboard for AQI Predictor (Pakistan - Top 6 Cities)
--------------------------------------------------------------------
Lets the user select a city, loads its latest features from the Feature
Store (BigQuery) and its model bundle from GCS (registered in the Vertex AI
Model Registry), then shows a 3-day AQI forecast with SHAP feature
importance and hazard alerts.

Run locally:  streamlit run web_app/app.py
Deployed:     Google Cloud Run (serverless) - authenticates automatically
              via the attached service account, no key file needed.
"""

import os
import sys
import joblib
import shap
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_pipeline.fetch_features import get_bq_client, get_table_ref, CITIES, FEATURE_COLS

load_dotenv()

st.set_page_config(page_title="Pearls AQI Predictor - Pakistan", page_icon="AQI", layout="wide")


GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCS_BUCKET = os.getenv("GCS_BUCKET")

AQI_BREAKPOINTS = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]


def classify_aqi(value: float):
    for low, high, label in AQI_BREAKPOINTS:
        if low <= value <= high:
            return label
    return "Out of range"


@st.cache_resource
def load_model_bundle(city: str):
    from google.cloud import storage

    local_path = f"/tmp/aqi_model_bundle_{city}.joblib"
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"aqi_forecast_{city}/aqi_model_bundle_{city}.joblib")
    blob.download_to_filename(local_path)
    return joblib.load(local_path)


@st.cache_data(ttl=600)
def load_latest_features(city: str):
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
    return df.sort_values("event_time")


def main():
    st.title("Pearls AQI Predictor - Pakistan")
    st.caption("3-day Air Quality Index forecast for major Pakistani cities, "
               "powered by Vertex AI (BigQuery Feature Store + Model Registry) and SHAP")

    city_display_names = {c: c.title() for c in CITIES}
    selected_display = st.selectbox(
        "Select a city",
        options=list(city_display_names.values()),
        index=0,
    )
    city = [k for k, v in city_display_names.items() if v == selected_display][0]

    with st.spinner(f"Loading latest data and model for {selected_display}..."):
        try:
            df = load_latest_features(city)
            bundle = load_model_bundle(city)
        except Exception as e:
            st.error(f"Could not load data/model for {selected_display}. Make sure "
                      f"pipelines have run at least once for this city. Error: {e}")
            return

    if df.empty:
        st.warning(f"No feature data yet for {selected_display}. "
                   f"Run feature_pipeline/fetch_features.py first.")
        return

    latest_row = df.iloc[-1]
    X_latest = pd.DataFrame([latest_row[FEATURE_COLS]])

    models = bundle["models"]
    scores = bundle["scores"]

    current_aqi = latest_row["aqi"]
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Current AQI ({selected_display})", f"{current_aqi:.0f}", classify_aqi(current_aqi))
    col2.metric("PM2.5", f"{latest_row['pm25']:.1f}")
    col3.metric("Last updated", latest_row["event_time"].strftime("%Y-%m-%d %H:%M"))

    st.subheader("Next 3 Days Forecast")
    forecast_vals = {}
    for horizon, model in models.items():
        pred = float(model.predict(X_latest)[0])
        forecast_vals[horizon] = pred

    forecast_df = pd.DataFrame({
        "Day": ["Today", "Day 1", "Day 2", "Day 3"],
        "AQI": [current_aqi, forecast_vals["day1"], forecast_vals["day2"], forecast_vals["day3"]],
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(forecast_df["Day"], forecast_df["AQI"], marker="o", linewidth=2, color="#d35400")
    ax.set_ylabel("AQI")
    ax.set_title(f"{selected_display} AQI Forecast")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Alerts")
    any_hazard = False
    for _, row in forecast_df.iterrows():
        if row["AQI"] >= 151:
            any_hazard = True
            st.error(f"{row['Day']}: AQI {row['AQI']:.0f} - {classify_aqi(row['AQI'])}. "
                      f"Sensitive groups should limit outdoor exposure.")
    if not any_hazard:
        st.success("No hazardous AQI levels predicted in the next 3 days.")

    with st.expander("Model Performance (on holdout data)"):
        perf_df = pd.DataFrame(scores).T
        st.dataframe(perf_df.style.format("{:.2f}"))

    st.subheader("What's driving the Day 1 prediction?")
    try:
        day1_model = models["day1"]
        explainer = shap.TreeExplainer(day1_model) if hasattr(day1_model, "estimators_") \
            else shap.LinearExplainer(day1_model, X_latest)
        shap_values = explainer.shap_values(X_latest)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_latest, plot_type="bar", show=False)
        st.pyplot(fig2)
    except Exception as e:
        st.info(f"SHAP explanation unavailable for this model type: {e}")

    st.caption("Data source: OpenWeather Air Pollution API | Feature Store: BigQuery | Model Registry: Vertex AI")


if __name__ == "__main__":
    main()
