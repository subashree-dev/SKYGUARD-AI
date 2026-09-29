import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# FILE PATHS
# ============================================================

clean_file = Path("data/clean/clean_baseline.csv")
synthetic_file = Path("data/synthetic/synthetic_sensor_data.csv")


# ============================================================
# LOAD DATA
# ============================================================

clean_df = pd.read_csv(clean_file)
test_df = pd.read_csv(synthetic_file)


# ============================================================
# FEATURES
# ============================================================

features = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa"
]

X_train = clean_df[features]
X_test = test_df[features]
y_test = test_df["is_anomaly"]


# ============================================================
# TRAIN MODEL ONLY ON CLEAN DATA
# ============================================================

model = IsolationForest(
    n_estimators=300,
    contamination=0.07,
    random_state=42
)

model.fit(X_train)


# ============================================================
# PREDICT SYNTHETIC TEST DATA
# ============================================================

predictions = model.predict(X_test)

ai_anomaly = [
    1 if value == -1 else 0
    for value in predictions
]


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    ai_anomaly
)

precision = precision_score(
    y_test,
    ai_anomaly,
    zero_division=0
)

recall = recall_score(
    y_test,
    ai_anomaly,
    zero_division=0
)

f1 = f1_score(
    y_test,
    ai_anomaly,
    zero_division=0
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n============================================================")
print("SKYGUARD AI - CLEAN TRAINING / FAULT TEST EVALUATION")
print("============================================================")

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"AI features      : {len(features)}")

print("\n------------------------------------------------------------")
print("EVALUATION RESULTS")
print("------------------------------------------------------------")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n------------------------------------------------------------")
print("CONFUSION MATRIX")
print("------------------------------------------------------------")

print(confusion_matrix(y_test, ai_anomaly))


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n------------------------------------------------------------")
print("CLASSIFICATION REPORT")
print("------------------------------------------------------------")

print(
    classification_report(
        y_test,
        ai_anomaly,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)


# ============================================================
# ANOMALY TYPE ANALYSIS
# ============================================================

test_df["ai_anomaly"] = ai_anomaly

print("\n------------------------------------------------------------")
print("DETECTION BY ANOMALY TYPE")
print("------------------------------------------------------------")

anomaly_types = test_df[test_df["is_anomaly"] == 1]

for anomaly_type, group in anomaly_types.groupby("anomaly_type"):

    detected = group["ai_anomaly"].sum()
    total = len(group)

    detection_rate = detected / total * 100

    print(
        f"{anomaly_type:20s} "
        f"{detected:2d}/{total:2d} detected "
        f"({detection_rate:.1f}%)"
    )


print("\n============================================================")
print("Evaluation complete.")
print("Training used only clean NOAA baseline data.")
print("Testing used NOAA data with synthetic sensor faults.")
print("============================================================")