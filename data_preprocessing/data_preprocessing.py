import pandas as pd


df = pd.read_csv("AAPL_stock_data.csv")

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
