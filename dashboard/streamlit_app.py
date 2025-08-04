import streamlit as st
import json
from PIL import Image
import os

st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("📈 Time Series Stock Forecasting")
st.write("Compare ARIMA, SARIMA, Prophet, and LSTM models on Apple Inc. stock prices.")

models = ["ARIMA", "SARIMA", "Prophet", "LSTM"]
selected_model = st.selectbox("Choose a model to view forecast & metrics:", models)

model_key = selected_model.lower()
metric_path = f"metrics/{model_key}_metrics.json"
plot_path = f"{model_key}_forecast.png"

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Forecast Plot")
    if os.path.exists(plot_path):
        st.image(Image.open(plot_path), use_container_width=True)  # ✅ Updated here
    else:
        st.warning("Forecast plot not found.")

with col2:
    st.subheader("📈 Evaluation Metrics")
    if os.path.exists(metric_path) and os.path.getsize(metric_path) > 0:
        with open(metric_path, "r") as f:
            try:
                metrics = json.load(f)
                st.metric("MAE", round(metrics["MAE"], 2))
                st.metric("RMSE", round(metrics["RMSE"], 2))
                st.metric("MAPE", f"{round(metrics['MAPE'], 2)}%")
            except json.JSONDecodeError:
                st.error("⚠ Invalid JSON format in metrics file.")
    else:
        st.warning("Metrics file not found or empty.")

with st.expander("📉 Show Seasonal Decomposition"):
    if os.path.exists("decomposition_plot.png"):
        st.image("decomposition_plot.png", caption="Additive Seasonal Decomposition", use_container_width=True)
    else:
        st.info("Run `decomposition_plot.py` to generate this.")
import pandas as pd

# Load all metrics
model_keys = ["arima", "sarima", "prophet", "lstm"]
metric_data = {}

for key in model_keys:
    path = f"metrics/{key}_metrics.json"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as f:
            try:
                metric_data[key.upper()] = json.load(f)
            except json.JSONDecodeError:
                continue

# Convert to DataFrame for comparison
df_metrics = pd.DataFrame(metric_data).T  # Transpose for model-wise rows

# Best model based on each metric
best_mae = df_metrics['MAE'].idxmin()
best_rmse = df_metrics['RMSE'].idxmin()
best_mape = df_metrics['MAPE'].idxmin()

st.subheader("📌 Model Inference Based on Metrics")

st.markdown(f"""
- ✅ **Best MAE**: {best_mae} with MAE = {df_metrics['MAE'].min():.2f}
- ✅ **Best RMSE**: {best_rmse} with RMSE = {df_metrics['RMSE'].min():.2f}
- ✅ **Best MAPE**: {best_mape} with MAPE = {df_metrics['MAPE'].min():.2f}%
""")

if best_mae == best_rmse == best_mape:
    st.success(f"🏆 Overall Best Performing Model: **{best_mae}**")
else:
    st.info("ℹ️ Different models perform better on different metrics. Choose based on your priority (e.g., lower % error or fewer outliers).")
