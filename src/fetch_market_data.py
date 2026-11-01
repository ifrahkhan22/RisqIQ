
import requests
import csv
from pathlib import Path

api_key = "your_key"
url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
params = {
    "api_key": api_key,
    "frequency": "daily",
    "data[0]": "value",
    "facets[series][]": "RBRTE",
    "start": "2015-01-01",
    "end": "2026-10-10",
    "sort[0][column]": "period",
    "sort[0][direction]": "asc",
    "length": 5000
}
response = requests.get(url, params=params, timeout=30)

print("HTTP status:", response.status_code)
response.raise_for_status()
result = response.json()
data = result.get("response", {}).get("data", [])

if not data:
    raise ValueError("No observations returned. Inspect the API response.")

print("Number of observations:", len(data))
print("Returned columns:", list(data[0].keys()))

output_folder = Path("data/raw")
output_folder.mkdir(parents=True, exist_ok=True)
output_file = output_folder / "brent_prices.csv"
with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

print("CSV saved to:", output_file.resolve())
print("Total observations saved:", len(data))