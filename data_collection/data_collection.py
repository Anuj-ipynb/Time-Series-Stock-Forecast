import yfinance as yf
import pandas as pd
import streamlit as st
# Download stock data (auto_adjust=True is now default)
@st.cache_data
def load_stock_data(ticker="AAPL"):
    data = yf.download(ticker, start="2010-01-01", end=None)
    return data

df = load_stock_data("AAPL")




# Only use available columns (no 'Adj Close')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

df.to_csv("AAPL_stock_data.csv")
print("✅ Data saved to AAPL_stock_data.csv")
