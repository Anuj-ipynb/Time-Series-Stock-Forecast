import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
import os
warnings.filterwarnings("ignore")

# Load and prepare data
df = pd.read_csv("AAPL_cleaned.csv", index_col=0)
df.index = pd.to_datetime(df.index)
monthly_df = df['Close'].resample('M').mean().dropna()  # ✅ FIXED LINE

# Train/test split
train = monthly_df.iloc[:-12]
test = monthly_df.iloc[-12:]

# Fit ARIMA on training data
model = ARIMA(train, order=(2, 1, 2))  
results = model.fit()

# Forecast 12 months
import sys
forecast_period = int(sys.argv[1]) if len(sys.argv) > 1 else 12  # default to 12 if not passed
forecast = results.forecast(steps=forecast_period)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(monthly_df, label='Actual')
plt.plot(forecast.index, forecast, label='ARIMA Forecast', linestyle='--')
plt.title("ARIMA Forecast")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("arima_forecast.png")

# Ensure the 'metrics' directory exists
os.makedirs("metrics", exist_ok=True)

# Evaluate
y_true = test.values
y_pred = forecast.values

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "MAPE": float(mape)
}

# Save metrics to JSON
with open("metrics/arima_metrics.json", "w") as f:
    json.dump(metrics, f)
