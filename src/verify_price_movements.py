
from pathlib import Path
import pandas as pd

# Locate the existing raw and processed files.
raw_file = Path("data/raw/brent_prices.csv")
processed_file = Path("data/processed/brent_market_features.csv")

for file_path in [raw_file, processed_file]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

raw = pd.read_csv(raw_file)
processed = pd.read_csv(processed_file)

raw["date"] = pd.to_datetime(raw["period"], errors="coerce")
raw["price_usd_per_barrel"] = pd.to_numeric(
    raw["value"], errors="coerce"
)

processed["date"] = pd.to_datetime(
    processed["date"], errors="coerce"
)
processed["price_usd_per_barrel"] = pd.to_numeric(
    processed["price_usd_per_barrel"], errors="coerce"
)
processed["daily_return_pct"] = pd.to_numeric(
    processed["daily_return_pct"], errors="coerce"
)

# Keep the necessary raw columns and sort chronologically.
raw = raw[["date", "price_usd_per_barrel"]]
raw = raw.sort_values("date").reset_index(drop=True)

# Recalculate returns independently from the raw prices.
raw["previous_available_price"] = (
    raw["price_usd_per_barrel"].shift(1)
)
raw["recomputed_return_pct"] = (
    raw["price_usd_per_barrel"]
    .pct_change(fill_method=None) * 100
)

processed_check = processed[
    ["date", "price_usd_per_barrel", "daily_return_pct"]
]

# Match observations by date and preserve unmatched dates.
comparison = raw.merge(
    processed_check,
    on="date",
    how="outer",
    suffixes=("_raw", "_processed"),
    indicator=True,
    validate="one_to_one"
)

date_mismatches = (comparison["_merge"] != "both").sum()
price_difference = (
    comparison["price_usd_per_barrel_raw"]
    - comparison["price_usd_per_barrel_processed"]
).abs()

price_mismatches = (
    comparison["_merge"].eq("both")
    & (price_difference > 1e-8)
).sum()

return_difference = (
    comparison["recomputed_return_pct"]
    - comparison["daily_return_pct"]
).abs()

return_mismatches = (
    comparison["_merge"].eq("both")
    & comparison["recomputed_return_pct"].notna()
    & comparison["daily_return_pct"].notna()
    & (return_difference > 1e-6)
).sum()

# Also check whether missing returns occur on different rows.
return_missingness_mismatches = (
    comparison["_merge"].eq("both")
    & (
        comparison["recomputed_return_pct"].isna()
        != comparison["daily_return_pct"].isna()
    )
).sum()

print("\n--- DATA VERIFICATION SUMMARY ---")
print("Raw rows:", len(raw))
print("Processed rows:", len(processed))
print("Dates missing from one file:", date_mismatches)
print("Price mismatches:", price_mismatches)
print("Return mismatches:", return_mismatches)
print(
    "Return missingness mismatches:",
    return_missingness_mismatches
)

# Display the largest independently recalculated movements.
matched = comparison[comparison["_merge"] == "both"].copy()
matched = matched[
    matched["recomputed_return_pct"].notna()
].copy()

matched["absolute_return_pct"] = (
    matched["recomputed_return_pct"].abs()
)

large_moves = matched[
    matched["absolute_return_pct"] >= 5
].sort_values("absolute_return_pct", ascending=False)

print("\nMovements with absolute returns of at least 5%:")
print("Count:", len(large_moves))

print("\nLargest 15 movements:")
print(
    large_moves[
        [
            "date",
            "previous_available_price",
            "price_usd_per_barrel_raw",
            "recomputed_return_pct",
            "daily_return_pct"
        ]
    ].head(15).to_string(index=False)
)