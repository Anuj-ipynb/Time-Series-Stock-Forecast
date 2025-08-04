import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("AAPL_cleaned.csv", index_col=0)
df.index = pd.to_datetime(df.index)
monthly = df['Close'].resample('M').mean().dropna()

result = seasonal_decompose(monthly, model='additive')
result.plot()
plt.tight_layout()
plt.savefig("decomposition_plot.png")
