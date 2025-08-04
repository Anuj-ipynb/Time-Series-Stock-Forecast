import yfinance as yf
import pandas as pd

# Download stock data (auto_adjust=True is now default)
df = yf.download("AAPL", start="2015-01-01", end="2024-01-01")

# Only use available columns (no 'Adj Close')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

df.to_csv("AAPL_stock_data.csv")
print("✅ Data saved to AAPL_stock_data.csv")
