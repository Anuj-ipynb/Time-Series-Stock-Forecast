# prophet_model.py
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings
import os
import json
import sys
import streamlit as st

warnings.filterwarnings("ignore")

# Accept dynamic forecast period
forecast_period = int(sys.argv[1]) if len(sys.argv) > 1 else 12

# Load and preprocess data
import yfinance as yf

@st.cache_data
def load_stock_data(ticker="AAPL"):
    data = yf.download(ticker, start="2010-01-01", end=None)
    return data

df = load_stock_data("AAPL")  # or pass user's selected ticker

df.index = pd.to_datetime(df.index)
monthly_df = df['Close'].resample('M').mean().dropna().reset_index()
monthly_df.columns = ['ds', 'y']

# Train/test split
train = monthly_df[:-forecast_period]
test = monthly_df[-forecast_period:]

# Train Prophet model
model = Prophet()
model.fit(train)

# Forecast future
future = model.make_future_dataframe(periods=forecast_period, freq='M')
forecast = model.predict(future)
forecast_subset = forecast[-forecast_period:]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(monthly_df['ds'], monthly_df['y'], label='Actual')
plt.plot(forecast_subset['ds'], forecast_subset['yhat'], label='Prophet Forecast', linestyle='--')
plt.title("Prophet Forecast")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("prophet_forecast.png")

# Evaluate
y_true = test['y'].values
y_pred = forecast_subset['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Save metrics
os.makedirs("metrics", exist_ok=True)
with open("metrics/prophet_metrics.json", "w") as f:
    json.dump({
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }, f)
