# --- lstm_model.py ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import json
import os
import warnings
import sys

warnings.filterwarnings("ignore")

# Accept dynamic forecast period
forecast_period = int(sys.argv[1]) if len(sys.argv) > 1 else 12

# Load data
df = pd.read_csv("AAPL_cleaned.csv", index_col=0, parse_dates=True)
monthly = df['Close'].resample('M').mean().dropna()

# Normalize
data = monthly.values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Create dataset
def create_dataset(dataset, time_step=12):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:i+time_step, 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 12
X, y = create_dataset(scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split
X_train = X[:-forecast_period]
X_test = X[-forecast_period:]
y_train = y[:-forecast_period]
y_test = y[-forecast_period:]

# Build and train model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=0)

# Predict
predictions = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = mean_squared_error(actual_prices, predicted_prices, squared=False)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

# Save metrics
os.makedirs("metrics", exist_ok=True)
metrics = {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}
with open("metrics/lstm_metrics.json", "w") as f:
    json.dump(metrics, f)

# Plot
full_index = monthly.index
plt.figure(figsize=(12, 6))
plt.plot(monthly, label="Actual")
plt.plot(full_index[-forecast_period:], predicted_prices, label="LSTM Forecast", linestyle='--')
plt.title("LSTM Forecast")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lstm_forecast.png")
plt.close()
