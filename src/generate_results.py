import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ==========================================
# FILE PATHS
# ==========================================

input_file = Path("data/synthetic/synthetic_sensor_data.csv")
model_file = Path("models/isolation_forest.pkl")
output_file = Path("data/synthetic/skyguard_results.csv")


# ==========================================
# LOAD DATA + MODEL
# ==========================================

df = pd.read_csv(input_file)

model = joblib.load(model_file)

print("Data and model loaded successfully!")


# ==========================================
# FEATURES
# ==========================================

features = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa"
]

X = df[features]


# ==========================================
# AI PREDICTION
# ==========================================

df["prediction"] = model.predict(X)

df["anomaly_score"] = model.decision_function(X)

df["ai_anomaly"] = (
    df["prediction"] == -1
).astype(int)


# ==========================================
# TRUST SCORE
# ==========================================

# Convert anomaly score into a simple 0-100 score
min_score = df["anomaly_score"].min()
max_score = df["anomaly_score"].max()

df["trust_score"] = (
    (df["anomaly_score"] - min_score)
    / (max_score - min_score)
    * 100
)

df["trust_score"] = df["trust_score"].round(1)


# ==========================================
# STATUS
# ==========================================

def get_status(row):

    if row["ai_anomaly"] == 1:

        if row["trust_score"] < 20:
            return "CRITICAL"

        elif row["trust_score"] < 40:
            return "WARNING"

        else:
            return "SUSPICIOUS"

    return "NORMAL"


df["status"] = df.apply(get_status, axis=1)


# ==========================================
# EXPLANATION
# ==========================================

def generate_explanation(row):

    if row["ai_anomaly"] == 0:
        return "Sensor readings appear consistent with the learned normal pattern."

    explanations = []

    if row["temperature_c"] > 45:
        explanations.append("temperature is unusually high")

    if row["temperature_c"] < 10:
        explanations.append("temperature is unusually low")

    if row["humidity_pct"] > 100:
        explanations.append("humidity is outside the normal physical range")

    if row["pressure_hpa"] < 980 or row["pressure_hpa"] > 1040:
        explanations.append("pressure is outside the expected range")

    if not explanations:
        return (
            "The combination of temperature, humidity and pressure "
            "differs significantly from the learned normal pattern."
        )

    return "Anomaly detected because " + " and ".join(explanations) + "."

df["explanation"] = df.apply(
    generate_explanation,
    axis=1
)


# ==========================================
# SAVE RESULTS
# ==========================================

df.to_csv(output_file, index=False)

print("\nSkyGuard results generated successfully!")

print("\nResult columns:")
print(list(df.columns))

print("\nStatus distribution:")
print(df["status"].value_counts())

print("\nSample results:")
print(
    df[
        [
            "timestamp",
            "temperature_c",
            "humidity_pct",
            "pressure_hpa",
            "trust_score",
            "status",
            "explanation"
        ]
    ].head(10)
)

print("\nSaved to:", output_file)