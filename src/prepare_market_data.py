
import pandas as pd
from pathlib import Path

input_file = Path("data/raw/brent_prices.csv")
df = pd.read_csv(input_file)
df["period"] = pd.to_datetime(df["period"], errors="raise")
df["value"] = pd.to_numeric(df["value"], errors="raise")
df = df.sort_values("period").reset_index(drop=True)

df = df.rename(columns={
    "period": "date",
    "value": "price_usd_per_barrel"
})

df["daily_return_pct"] = (
    df["price_usd_per_barrel"].pct_change(fill_method=None) * 100
)

# Calculate the 20-observation rolling standard deviation
rolling_std = df["daily_return_pct"].rolling(window=20).std()

# Annualize volatility and express it as a percentage
df["annualized_volatility_20d_pct"] = rolling_std * (252 ** 0.5)

# Typical daily move over the previous 20 days (shift(1) leaves out today)
df["prev_20d_std"] = df["daily_return_pct"].rolling(20).std().shift(1)

# How many "typical moves" was today's move?
df["shock_score"] = df["daily_return_pct"].abs() / df["prev_20d_std"]

# Two definitions: relative to recent calm, and the simple 5% rule
df["vol_shock"] = df["shock_score"] >= 2.5
df["large_move_5pct"] = df["daily_return_pct"].abs() >= 5

print("Vol-adjusted shocks:", int(df["vol_shock"].sum()))
print("5% moves:", int(df["large_move_5pct"].sum()))

output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)
output_file = output_folder / "brent_market_features.csv"
df.to_csv(output_file, index=False)

print("Rows prepared:", len(df))
print("Earliest date:", df["date"].min().date())
print("Latest date:", df["date"].max().date())
print("Price data type:", df["price_usd_per_barrel"].dtype)
print("Missing daily returns:", df["daily_return_pct"].isna().sum())
print("Missing rolling volatility:",
      df["annualized_volatility_20d_pct"].isna().sum())

print("\nLatest 5 observations:")
print(df[[
    "date",
    "price_usd_per_barrel",
    "daily_return_pct",
    "annualized_volatility_20d_pct"
]].tail().to_string(index=False))

print("\nSaved to:", output_file.resolve())