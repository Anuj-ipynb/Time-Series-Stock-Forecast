import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf
import streamlit as st
@st.cache_data
def load_stock_data(ticker="AAPL"):
    data = yf.download(ticker, start="2010-01-01", end=None)
    return data

df = load_stock_data("AAPL")
df.index = pd.to_datetime(df.index)
monthly = df['Close'].resample('M').mean().dropna()

result = seasonal_decompose(monthly, model='additive')
result.plot()
plt.tight_layout()
plt.savefig("decomposition_plot.png")
