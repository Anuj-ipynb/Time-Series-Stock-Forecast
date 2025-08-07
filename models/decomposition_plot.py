# decomposition_plot.py
import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# Read inputs
forecast_period = int(sys.argv[1]) if len(sys.argv) > 1 else 12
ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"

# Download & preprocess
df = yf.download(ticker, start="2010-01-01")
monthly = df["Close"].resample("M").mean().dropna()

# Slice last forecast_period months
if len(monthly) < forecast_period:
    print("Not enough data to slice for seasonal decomposition.")
    exit()

last_period = monthly[-forecast_period:]

# Ensure at least 2*period points
if len(last_period) < 24:
    print("Too few points for decomposition.")
    exit()

# Decompose and plot
result = seasonal_decompose(last_period, model="additive", period=12)
fig = result.plot()
fig.set_size_inches(12, 8)
plt.suptitle(f"{ticker} - Seasonal Decomposition (Last {forecast_period} Months)", fontsize=14)
plt.tight_layout()
plt.savefig("decomposition_plot.png")
plt.close()
