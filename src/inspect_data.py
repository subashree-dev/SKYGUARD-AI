import pandas as pd
from pathlib import Path

input_file = Path("data/raw/GHCNh_INM00043278_2026.psv")
output_file = Path("data/clean/clean_baseline.csv")

df = pd.read_csv(input_file, sep="|")

clean_df = df[
    [
        "STATION",
        "DATE",
        "temperature",
        "relative_humidity",
        "station_level_pressure"
    ]
].copy()

clean_df.columns = [
    "station_id",
    "timestamp",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa"
]

clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])

clean_df = clean_df.dropna(
    subset=["temperature_c", "humidity_pct", "pressure_hpa"]
)

clean_df = clean_df.sort_values("timestamp")

clean_df.to_csv(output_file, index=False)

print("Clean dataset created successfully!")
print("Rows:", len(clean_df))
print("Columns:", list(clean_df.columns))
print("\nFirst 5 rows:")
print(clean_df.head())

print("\nSaved to:", output_file)