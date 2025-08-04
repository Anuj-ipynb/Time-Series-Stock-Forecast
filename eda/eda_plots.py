import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import streamlit as st
import yfinance as yf
@st.cache_data
def load_stock_data(ticker="AAPL"):
    data = yf.download(ticker, start="2010-01-01", end=None)
    return data

df = load_stock_data("AAPL")
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
