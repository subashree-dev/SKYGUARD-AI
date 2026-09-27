import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# FILE PATHS
# ==========================================

input_file = Path("data/synthetic/synthetic_sensor_data.csv")
model_file = Path("models/isolation_forest.pkl")


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(input_file)

print("Dataset loaded!")
print("Rows:", len(df))


# ==========================================
# FEATURES USED BY THE AI
# ==========================================

features = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa"
]

X = df[features]


# ==========================================
# TRAIN ISOLATION FOREST
# ==========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.07,
    random_state=42
)

model.fit(X)


# ==========================================
# PREDICT ANOMALIES
# ==========================================

predictions = model.predict(X)

# Isolation Forest:
#  1  = normal
# -1  = anomaly

df["ai_prediction"] = predictions

df["ai_anomaly"] = df["ai_prediction"].apply(
    lambda x: 1 if x == -1 else 0
)


# ==========================================
# EVALUATION
# ==========================================

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        df["is_anomaly"],
        df["ai_anomaly"]
    )
)

print("\nClassification Report:")
print(
    classification_report(
        df["is_anomaly"],
        df["ai_anomaly"],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, model_file)

print("\nModel saved successfully!")
print("Saved to:", model_file)