import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("AAPL_cleaned.csv", index_col=0)
df.index = pd.to_datetime(df.index)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.title("AAPL Closing Price")
plt.grid(True)
plt.legend()
plt.savefig("close_price_plot.png")

# ADF Test
monthly = df['Close'].resample('M').mean()
result = adfuller(monthly.dropna())
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
