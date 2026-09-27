import pandas as pd
from pathlib import Path


# ============================================================
# LOAD SKYGUARD DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "synthetic"
    / "skyguard_results.csv"
)

df = pd.read_csv(DATA_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])


# ============================================================
# BASIC CALCULATIONS
# ============================================================

total_readings = len(df)

ai_anomalies = int(df["ai_anomaly"].sum())

normal_readings = total_readings - ai_anomalies

anomaly_rate = (
    ai_anomalies / total_readings * 100
    if total_readings > 0
    else 0
)

critical_count = int(
    (df["status"] == "CRITICAL").sum()
)

warning_count = int(
    (df["status"] == "WARNING").sum()
)

suspicious_count = int(
    (df["status"] == "SUSPICIOUS").sum()
)

average_trust = df["trust_score"].mean()

latest = df.iloc[-1]

latest_temperature = latest["temperature_c"]

latest_humidity = latest["humidity_pct"]

latest_pressure = latest["pressure_hpa"]

latest_trust = latest["trust_score"]

latest_status = latest["status"]

latest_explanation = latest["explanation"]