import streamlit as st
import json
from PIL import Image
import os
import subprocess
import pandas as pd
import yfinance as yf

# --- Streamlit Page Config ---
st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("📈 Time Series Stock Forecasting")
st.write("Compare ARIMA, SARIMA, Prophet, and LSTM models on stock prices.")

# --- Ticker Input + Load Data ---
ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT):", value="AAPL")

@st.cache_data
def load_stock_data(ticker="AAPL"):
    return yf.download(ticker, start="2010-01-01")

df = load_stock_data(ticker)

if df.empty:
    st.error("⚠ Failed to load data. Check ticker symbol or internet connection.")

# --- Forecast Period Slider ---
forecast_period = st.slider("Forecast Months", 3, 24, 12)

# --- Run selected model script BEFORE showing metrics/plots ---
models = ["ARIMA", "SARIMA", "Prophet", "LSTM"]
selected_model = st.selectbox("Choose a model to view forecast & metrics:", models)

if selected_model == "ARIMA":
    subprocess.run(["python", "models/arima_model.py", str(forecast_period)])
elif selected_model == "LSTM":
    subprocess.run(["python", "models/lstm_model.py", str(forecast_period)])
elif selected_model == "Prophet":
    subprocess.run(["python", "models/prophet_model.py", str(forecast_period)])
elif selected_model == "SARIMA":
    subprocess.run(["python", "models/sarima_model.py", str(forecast_period)])

# --- Run seasonal decomposition script with ticker & forecast_period ---
subprocess.run(["python", "decomposition_plot.py", str(forecast_period), ticker])

# --- Forecast Plot + Metrics Display ---
model_key = selected_model.lower()
metric_path = f"metrics/{model_key}_metrics.json"
plot_path = f"{model_key}_forecast.png"

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Forecast Plot")
    if os.path.exists(plot_path):
        st.image(Image.open(plot_path), use_container_width=True)
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
