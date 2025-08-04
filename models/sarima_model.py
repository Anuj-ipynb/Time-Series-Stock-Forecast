import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

warnings.filterwarnings("ignore")

df = pd.read_csv("AAPL_cleaned.csv", index_col=0)
df.index = pd.to_datetime(df.index)
monthly_df = df['Close'].resample('M').mean().dropna()

# Split data
train = monthly_df[:-12]
test = monthly_df[-12:]

# Fit SARIMA model (adjust seasonal_order as needed)
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)

import sys
forecast_period = int(sys.argv[1]) if len(sys.argv) > 1 else 12  # default to 12 if not passed
forecast = results.forecast(steps=forecast_period)


# Plot
plt.figure(figsize=(12, 6))
plt.plot(monthly_df, label='Actual')
plt.plot(forecast.index, forecast, label='SARIMA Forecast', linestyle='--')
plt.title("SARIMA Forecast")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sarima_forecast.png")

# Evaluate
y_true = test.values
y_pred = forecast.values

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

os.makedirs("metrics", exist_ok=True)
with open("metrics/sarima_metrics.json", "w") as f:
    json.dump({"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}, f)
