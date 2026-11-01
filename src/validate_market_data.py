
import csv
from pathlib import Path

file_path = Path("data/raw/brent_prices.csv")
with file_path.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    data = list(reader)

print("Number of rows:", len(data))

dates = [row["period"] for row in data if row["period"]]
print("Earliest date:", min(dates) if dates else "No dates found")
print("Latest date:", max(dates) if dates else "No dates found")

missing_prices = sum(
    1 for row in data
    if not row["value"] or not row["value"].strip()
)

print("Missing price values:", missing_prices)
duplicate_dates = len(dates) - len(set(dates))
print("Duplicate dates:", duplicate_dates)
print("\nFirst 3 records:")
for row in data[:3]:
    print(row)