import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings
import os
import json

warnings.filterwarnings("ignore")

df = pd.read_csv("AAPL_cleaned.csv", index_col=0)
df.index = pd.to_datetime(df.index)
monthly_df = df['Close'].resample('M').mean().dropna().reset_index()
monthly_df.columns = ['ds', 'y']

# Train/test split
train = monthly_df[:-12]
test = monthly_df[-12:]

model = Prophet()
model.fit(train)

future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Extract only the forecasted part
forecast_subset = forecast[-12:]

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
rmse = mean_squared_error(y_true, y_pred, squared= False)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

os.makedirs("metrics", exist_ok=True)
with open("metrics/prophet_metrics.json", "w") as f:
    json.dump({"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}, f)
