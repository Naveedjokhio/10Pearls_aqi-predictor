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
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_pipeline.fetch_features import get_bq_client, get_table_ref, CITIES, FEATURE_COLS
# Imported (even though not called directly) so that joblib can resolve
# this class by name when unpickling a model bundle that contains a
# NeuralNet model - without this import, unpickling fails with
# "Can't get attribute 'KerasMLPRegressor'".
from training_pipeline.model_wrapper import KerasMLPRegressor  # noqa: F401

load_dotenv()

st.set_page_config(page_title="Pearls AQI Predictor - Pakistan", page_icon="🌫️", layout="wide")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCS_BUCKET = os.getenv("GCS_BUCKET")

# (low, high, label, color)
AQI_BANDS = [
    (0, 50, "Good", "#4CAF50"),
    (51, 100, "Moderate", "#FFC107"),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF9800"),
    (151, 200, "Unhealthy", "#F44336"),
    (201, 300, "Very Unhealthy", "#9C27B0"),
    (301, 500, "Hazardous", "#7B241C"),
]


def classify_aqi(value: float):
    for low, high, label, color in AQI_BANDS:
        if low <= value <= high:
            return label, color
    return "Out of range", "#777777"


CUSTOM_CSS = """
<style>
    .aqi-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        color: white;
    }
    .metric-card {
        background-color: #1C1F26;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2A2E37;
    }
    .metric-label {
        color: #9AA0AC;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FAFAFA;
    }
</style>
"""


@st.cache_resource
def load_model_bundle(city: str):
    from google.cloud import storage

    local_path = f"/tmp/aqi_model_bundle_{city}.joblib"
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"aqi_forecast_{city}/model.joblib")
    blob.download_to_filename(local_path)
    return joblib.load(local_path)


@st.cache_data(ttl=600)
def load_latest_features(city: str):
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
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


@st.cache_data(ttl=600)
def load_all_cities_snapshot():
    """One current AQI reading per city, for the overview strip at the top.
    Uses a single lightweight query (latest row per city) instead of
    fetching each city's full history."""
    client = get_bq_client()
    query = f"""
        SELECT city, aqi
        FROM (
            SELECT city, aqi,
                   ROW_NUMBER() OVER (PARTITION BY city ORDER BY event_time DESC) AS rn
            FROM `{get_table_ref()}`
        )
        WHERE rn = 1
    """
    try:
        return client.query(query).to_dataframe()
    except Exception:
        return pd.DataFrame(columns=["city", "aqi"])


def render_city_overview_strip(current_city: str):
    snapshot = load_all_cities_snapshot()
    if snapshot.empty:
        return
    cols = st.columns(len(CITIES))
    for col, city in zip(cols, CITIES):
        match = snapshot[snapshot["city"] == city]
        with col:
            if not match.empty:
                aqi_val = match.iloc[0]["aqi"]
                label, color = classify_aqi(aqi_val)
                highlight = "border: 2px solid #FF5722;" if city == current_city else "border: 1px solid #2A2E37;"
                st.markdown(
                    f"""
                    <div style="background-color:#1C1F26; border-radius:10px; padding:12px; text-align:center; {highlight}">
                        <div style="font-size:0.75rem; color:#9AA0AC; text-transform:uppercase;">{city.title()}</div>
                        <div style="font-size:1.6rem; font-weight:700; color:{color};">{aqi_val:.0f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div style="background-color:#1C1F26; border-radius:10px; padding:12px; text-align:center;">
                    <div style="font-size:0.75rem; color:#9AA0AC;">{city.title()}</div>
                    <div style="font-size:1.2rem; color:#555;">--</div></div>""",
                    unsafe_allow_html=True,
                )


def render_forecast_chart(city_display: str, forecast_df: pd.DataFrame):
    colors = [classify_aqi(v)[1] for v in forecast_df["AQI"]]
    # Keep the last point's label anchored to its left so it can't be
    # clipped by the right edge of the chart.
    n = len(forecast_df)
    text_positions = ["top center"] * (n - 1) + ["middle left"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df["Day"], y=forecast_df["AQI"],
        mode="lines+markers+text",
        line=dict(color="#FF5722", width=3),
        marker=dict(size=14, color=colors, line=dict(width=2, color="white")),
        text=[f"{v:.0f}" for v in forecast_df["AQI"]],
        textposition=text_positions,
        textfont=dict(size=14, color="#FAFAFA"),
        fill="tozeroy",
        fillcolor="rgba(255,87,34,0.08)",
        cliponaxis=False,
    ))
    fig.update_layout(
        title=f"{city_display} AQI Forecast",
        template="plotly_dark",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        height=380,
        margin=dict(t=60, b=40, l=40, r=40),
        yaxis_title="AQI",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap_chart(shap_values, feature_names):
    import numpy as np
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = mean_abs.argsort()

    fig = go.Figure(go.Bar(
        x=mean_abs[order],
        y=[feature_names[i] for i in order],
        orientation="h",
        marker=dict(color="#FF5722"),
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        height=380,
        margin=dict(t=20, b=40, l=20, r=20),
        xaxis_title="mean(|SHAP value|)",
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("🌫️ Pearls AQI Predictor — Pakistan")
    st.caption("3-day Air Quality Index forecast for major Pakistani cities, "
               "powered by Vertex AI (BigQuery Feature Store + Model Registry) and SHAP")

    city_display_names = {c: c.title() for c in CITIES}
    selected_display = st.selectbox(
        "Select a city",
        options=list(city_display_names.values()),
        index=0,
    )
    city = [k for k, v in city_display_names.items() if v == selected_display][0]

    st.markdown("##### All Cities — Current AQI")
    render_city_overview_strip(city)
    st.divider()

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
    label, color = classify_aqi(current_aqi)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current AQI — {selected_display}</div>
            <div class="metric-value">{current_aqi:.0f}</div>
            <span class="aqi-badge" style="background-color:{color};">{label}</span>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PM2.5</div>
            <div class="metric-value">{latest_row['pm25']:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Last Updated</div>
            <div class="metric-value" style="font-size:1.3rem;">{latest_row['event_time'].strftime('%b %d, %H:%M')}</div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.subheader("📈 Next 3 Days Forecast")
    forecast_vals = {}
    for horizon, model in models.items():
        pred = float(model.predict(X_latest)[0])
        forecast_vals[horizon] = pred

    forecast_df = pd.DataFrame({
        "Day": ["Today", "Day 1", "Day 2", "Day 3"],
        "AQI": [current_aqi, forecast_vals["day1"], forecast_vals["day2"], forecast_vals["day3"]],
    })
    render_forecast_chart(selected_display, forecast_df)

    st.subheader("⚠️ Alerts")
    any_hazard = False
    for _, row in forecast_df.iterrows():
        row_label, row_color = classify_aqi(row["AQI"])
        if row["AQI"] >= 151:
            any_hazard = True
            st.markdown(
                f"""<div style="background-color:rgba(244,67,54,0.12); border-left:4px solid {row_color};
                border-radius:6px; padding:10px 16px; margin-bottom:8px;">
                <b>{row['Day']}</b>: AQI {row['AQI']:.0f} — {row_label}. Sensitive groups should limit outdoor exposure.
                </div>""", unsafe_allow_html=True)
    if not any_hazard:
        st.success("No hazardous AQI levels predicted in the next 3 days.")

    with st.expander("📊 Model Performance (on holdout data)"):
        perf_df = pd.DataFrame(scores).T
        st.dataframe(perf_df.style.format("{:.2f}"), use_container_width=True)

    st.subheader("🔍 What's driving the Day 1 prediction?")
    try:
        day1_model = models["day1"]
        if hasattr(day1_model, "estimators_"):
            explainer = shap.TreeExplainer(day1_model)
            shap_values = explainer.shap_values(X_latest)
        elif hasattr(day1_model, "coef_"):
            explainer = shap.LinearExplainer(day1_model, X_latest)
            shap_values = explainer.shap_values(X_latest)
        else:
            # Neural net or any other model type without a specialized
            # SHAP explainer - fall back to the general-purpose KernelExplainer.
            explainer = shap.KernelExplainer(day1_model.predict, X_latest)
            shap_values = explainer.shap_values(X_latest, nsamples=100)
        render_shap_chart(shap_values, FEATURE_COLS)
    except Exception as e:
        st.info(f"SHAP explanation unavailable for this model type: {e}")

    st.caption("Data source: OpenWeather Air Pollution API | Feature Store: BigQuery | Model Registry: Vertex AI")


if __name__ == "__main__":
    main()