import pandas as pd
import numpy as np

# ==========================
# Load Dataset
# ==========================
file_path = "BTCUSDT-1h-2023-01.csv"

columns = [
    "Open Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close Time",
    "Quote Asset Volume",
    "Number of Trades",
    "Taker Buy Base Asset Volume",
    "Taker Buy Quote Asset Volume",
    "Ignore"
]

df = pd.read_csv(file_path, header=None)
df.columns = columns

# ==========================
# Convert Official Binance UTC Timestamps
# ==========================
df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms", utc=True)

# ==========================
# Remove Ignore Column
# ==========================
df.drop(columns=["Ignore"], inplace=True)

# ==========================
# Convert Numeric Columns
# ==========================
numeric_cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Quote Asset Volume",
    "Number of Trades",
    "Taker Buy Base Asset Volume",
    "Taker Buy Quote Asset Volume"
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

# ==========================
# Candle Features
# ==========================
df["Price Change"] = df["Close"] - df["Open"]

df["Return (%)"] = (
    (df["Close"] - df["Open"]) / df["Open"]
) * 100

df["Body Size"] = abs(df["Close"] - df["Open"])

df["Upper Wick"] = (
    df["High"] - df[["Open", "Close"]].max(axis=1)
)

df["Lower Wick"] = (
    df[["Open", "Close"]].min(axis=1) - df["Low"]
)

df["Volatility"] = df["High"] - df["Low"]

# ==========================
# Lag Features
# ==========================
df["Close_Lag_1"] = df["Close"].shift(1)
df["Close_Lag_2"] = df["Close"].shift(2)
df["Close_Lag_3"] = df["Close"].shift(3)

df["Volume_Lag_1"] = df["Volume"].shift(1)
df["Volume_Lag_2"] = df["Volume"].shift(2)

# ==========================
# Simple Moving Averages
# ==========================
df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

# ==========================
# Exponential Moving Averages
# ==========================
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

# ==========================
# RSI (14 Period)
# ==========================
delta = df["Close"].diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss

df["RSI_14"] = 100 - (100 / (1 + rs))

# ==========================
# MACD
# ==========================
df["MACD"] = df["EMA_12"] - df["EMA_26"]

df["MACD_Signal"] = (
    df["MACD"]
    .ewm(span=9, adjust=False)
    .mean()
)

df["MACD_Histogram"] = (
    df["MACD"] - df["MACD_Signal"]
)

# ==========================
# Bollinger Bands (20 Period)
# ==========================
rolling_mean = (
    df["Close"]
    .rolling(window=20)
    .mean()
)

rolling_std = (
    df["Close"]
    .rolling(window=20)
    .std()
)

df["BB_Middle"] = rolling_mean
df["BB_Upper"] = rolling_mean + (2 * rolling_std)
df["BB_Lower"] = rolling_mean - (2 * rolling_std)

# ==========================
# Target Variable
# Predict Next Candle Close
# ==========================
df["Target_Close"] = df["Close"].shift(-1)

# ==========================
# Drop Rows with Missing Values
# ==========================
df.dropna(inplace=True)

# ==========================
# Save Final Dataset
# ==========================
df.to_csv(
    "BTCUSDT_Enriched.csv",
    index=False
)

df.to_excel(
    "BTCUSDT_Enriched.xlsx",
    index=False
)

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nPreview:")
print(df.head())