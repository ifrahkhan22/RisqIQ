
import pandas as pd

# Load the prepared Brent price dataset
file_path = "data/processed/brent_market_features.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

df["abs_return_pct"] = df["daily_return_pct"].abs()
df["large_movement"] = df["abs_return_pct"] >= 5

valid_returns = df.dropna(subset=["daily_return_pct"]).copy()

# Count large movements across the full dataset
large_count = int(valid_returns["large_movement"].sum())
total_returns = len(valid_returns)

print("Total valid daily returns:", total_returns)
print("Movements of at least 5%:", large_count)
print(
    "Percentage of returns with large movements:",
    round(large_count / total_returns * 100, 2),
    "%"
)

valid_returns["month"] = (
    valid_returns["date"].dt.to_period("M")
)

monthly = (
    valid_returns.groupby("month")
    .agg(
        total_observations=("daily_return_pct", "count"),
        large_movements=("large_movement", "sum")
    )
    .reset_index()
)
monthly["large_movement_pct"] = (
    monthly["large_movements"]
    / monthly["total_observations"] * 100
)

# Display months with the most large movements
monthly_ranked = monthly.sort_values(
    "large_movements", ascending=False
)

print("\nTop 10 months by number of large movements:")
print(monthly_ranked.head(10).to_string(index=False))

# Identify clusters of consecutive large observations
# A new cluster begins when the current observation is not large,
# or the previous observation was not large.
valid_returns["new_cluster"] = (
    valid_returns["large_movement"]
    & ~valid_returns["large_movement"].shift(
        1, fill_value=False
    )
)

valid_returns["cluster_id"] = (
    valid_returns["new_cluster"].cumsum()
)

clusters = (
    valid_returns[valid_returns["large_movement"]]
    .groupby("cluster_id")
    .agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        large_observations=("date", "count")
    )
    .reset_index(drop=True)
)

# Keep only clusters with at least two consecutive observations
clusters = clusters[clusters["large_observations"] >= 2]

print("\nClusters with at least 2 consecutive large observations:")
if clusters.empty:
    print("No consecutive clusters found.")
else:
    print(clusters.to_string(index=False))