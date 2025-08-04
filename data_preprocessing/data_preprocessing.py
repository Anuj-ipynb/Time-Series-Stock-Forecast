import pandas as pd
import streamlit as st
import yfinance as yf

@st.cache_data
def load_stock_data(ticker="AAPL"):
    data = yf.download(ticker, start="2010-01-01", end=None)
    return data

df = load_stock_data("AAPL")

if 'Unnamed: 0' in df.columns:
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
elif 'Date' not in df.columns:
    # Try to detect the first column name if it's not 'Date'
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df = df.dropna(subset=['Date'])

df.set_index('Date', inplace=True)

expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df = df[[col for col in expected_columns if col in df.columns]]

# ✅ Step 7: Save cleaned data
df.to_csv("AAPL_cleaned.csv")
print("✅ AAPL_cleaned.csv saved successfully.")
