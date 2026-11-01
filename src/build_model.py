import pandas as pd

df = pd.read_csv("data/processed/brent_market_features.csv", parse_dates=["date"])
s = df["vol_shock"].astype(int)

# Target: is there a shock on ANY of the next 5 trading days?
df["target_next5"] = s.rolling(5).max().shift(-5)
big = (df["daily_return_pct"].abs() >= 4).astype(int)
df["target_turbulence"] = big.rolling(5).max().shift(-5)

# Features: only information known up to today
df["abs_return_pct"] = df["daily_return_pct"].abs()
df["abs_return_5d_pct"] = df["daily_return_pct"].rolling(5).sum().abs()
df["vol_5d"] = df["daily_return_pct"].rolling(5).std()
df["vol_20d"] = df["daily_return_pct"].rolling(20).std()
df["vol_60d"] = df["daily_return_pct"].rolling(60).std()
df["vol_20d_vs_1y"] = df["vol_20d"] / df["vol_20d"].rolling(250).median()
df["vol_ratio"] = df["vol_5d"] / df["vol_60d"]   # is volatility rising?
df["shocks_last_5d"] = s.rolling(5).sum()
df["shocks_last_20d"] = s.rolling(20).sum()
df["price_vs_ma50_pct"] = (
    df["price_usd_per_barrel"] / df["price_usd_per_barrel"].rolling(50).mean() - 1
) * 100

features = ["abs_return_pct", "abs_return_5d_pct", "vol_5d", "vol_20d", "vol_60d","vol_20d_vs_1y",
            "vol_ratio", "shocks_last_5d", "shocks_last_20d", "price_vs_ma50_pct"]

# Drop rows where something is missing (start of data, last 5 days)
df = df.dropna(subset=features + ["target_next5", "target_turbulence"]).copy()
df[["date"] + features + ["target_next5", "target_turbulence"]].to_csv(
    "data/processed/model_dataset.csv", index=False)
print("Turbulence rate:", round(df["target_turbulence"].mean() * 100, 1), "%")

print("Rows:", len(df))
print("Share of days with a shock coming in the next 5 days:",
      round(df["target_next5"].mean() * 100, 1), "%")