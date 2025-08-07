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
# --- Exploratory Data Analysis (EDA) ---
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

st.subheader("🔍 Exploratory Data Analysis (EDA)")

# Plot closing prices
if not df.empty:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['Close'], label='Close Price')
    ax.set_title(f"{ticker.upper()} Closing Price")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ADF Test
    monthly = df['Close'].resample('M').mean()
    result = adfuller(monthly.dropna())

    st.markdown("**Augmented Dickey-Fuller (ADF) Test**")
    st.markdown(f"""
    - **ADF Statistic**: `{result[0]:.4f}`
    - **p-value**: `{result[1]:.4f}`
    - **Stationarity**: {"✅ Likely Stationary" if result[1] < 0.05 else "❌ Likely Non-Stationary"}
    """)
else:
    st.warning("No data available to perform EDA.")
# --- Best Model Comparison Section ---
st.subheader("🏆 Best Model Comparison")

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


if metric_data:
    df_metrics = pd.DataFrame(metric_data).T 
    st.dataframe(df_metrics.style.format({
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "MAPE": "{:.2f}"
    }), use_container_width=True)

    # Identify best models
    best_mae = df_metrics['MAE'].idxmin()
    best_rmse = df_metrics['RMSE'].idxmin()
    best_mape = df_metrics['MAPE'].idxmin()

    st.markdown(f"""
    - ✅ **Best MAE**: `{best_mae}` with MAE = `{df_metrics['MAE'].min():.2f}`
    - ✅ **Best RMSE**: `{best_rmse}` with RMSE = `{df_metrics['RMSE'].min():.2f}`
    - ✅ **Best MAPE**: `{best_mape}` with MAPE = `{df_metrics['MAPE'].min():.2f}%`
    """)

    if best_mae == best_rmse == best_mape:
        st.success(f"🏆 Overall Best Performing Model: **{best_mae}**")
    else:


