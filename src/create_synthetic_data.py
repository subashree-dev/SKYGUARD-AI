import pandas as pd
import numpy as np
from pathlib import Path

# Input and output files
input_file = Path("data/clean/clean_baseline.csv")
output_file = Path("data/synthetic/synthetic_sensor_data.csv")

# Load clean baseline data
df = pd.read_csv(input_file)

# Labels
df["is_anomaly"] = 0
df["anomaly_type"] = "normal"

# Make results reproducible
np.random.seed(42)

# ==========================================
# 1. TEMPERATURE SPIKE
# ==========================================

spike_indices = np.random.choice(
    df.index,
    size=15,
    replace=False
)

df.loc[spike_indices, "temperature_c"] += np.random.choice(
    [15, 20, 25],
    size=len(spike_indices)
)

df.loc[spike_indices, "is_anomaly"] = 1
df.loc[spike_indices, "anomaly_type"] = "temperature_spike"


# ==========================================
# 2. STUCK TEMPERATURE SENSOR
# ==========================================

stuck_start = 300
stuck_end = 315

stuck_value = df.loc[stuck_start, "temperature_c"]

df.loc[
    stuck_start:stuck_end,
    "temperature_c"
] = stuck_value

df.loc[
    stuck_start:stuck_end,
    "is_anomaly"
] = 1

df.loc[
    stuck_start:stuck_end,
    "anomaly_type"
] = "temperature_stuck"


# ==========================================
# 3. TEMPERATURE DRIFT
# ==========================================

drift_start = 600
drift_end = 630

drift_values = np.linspace(
    0,
    12,
    drift_end - drift_start + 1
)

df.loc[
    drift_start:drift_end,
    "temperature_c"
] += drift_values

df.loc[
    drift_start:drift_end,
    "is_anomaly"
] = 1

df.loc[
    drift_start:drift_end,
    "anomaly_type"
] = "temperature_drift"


# ==========================================
# 4. HUMIDITY SPIKE
# ==========================================

humidity_indices = np.random.choice(
    df.index,
    size=15,
    replace=False
)

df.loc[
    humidity_indices,
    "humidity_pct"
] += 15

df.loc[
    humidity_indices,
    "is_anomaly"
] = 1

df.loc[
    humidity_indices,
    "anomaly_type"
] = "humidity_spike"


# ==========================================
# SAVE DATASET
# ==========================================

df.to_csv(output_file, index=False)

print("Synthetic dataset created successfully!")
print("Rows:", len(df))
print("Columns:", list(df.columns))

print("\nAnomaly counts:")
print(df["anomaly_type"].value_counts())

print("\nSaved to:", output_file)
