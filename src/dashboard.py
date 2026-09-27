import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🌤️",
    layout="wide"
)


# ==========================================
# LOAD DATA
# ==========================================

data_file = Path("data/synthetic/skyguard_results.csv")

df = pd.read_csv(data_file)


# ==========================================
# HEADER
# ==========================================

st.title("🌤️ SkyGuard AI")

st.subheader(
    "Intelligent Anomaly Detection for Automatic Weather Stations"
)

st.write(
    "AI-powered monitoring of temperature, humidity and pressure "
    "sensor readings."
)

st.divider()


# ==========================================
# SELECT SENSOR READING
# ==========================================

st.subheader("🔎 Sensor Reading Analysis")

# ==========================================
# READING SELECTION
# ==========================================

anomaly_indices = df.index[df["ai_anomaly"] == 1].tolist()

col_select, col_button = st.columns([3, 1])

with col_select:
    selected_index = st.slider(
        "Select a sensor reading",
        min_value=0,
        max_value=len(df) - 1,
        value=len(df) - 1
    )

with col_button:
    st.write("")
    st.write("")

    if st.button("🚨 Show Anomaly"):
        if anomaly_indices:
            selected_index = anomaly_indices[0]

selected = df.iloc[selected_index]

temperature = selected["temperature_c"]
humidity = selected["humidity_pct"]
pressure = selected["pressure_hpa"]
trust_score = selected["trust_score"]
status = selected["status"]
explanation = selected["explanation"]
timestamp = selected["timestamp"]

st.caption(f"Reading timestamp: {timestamp}")

# ==========================================
# SENSOR METRICS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🌡️ Temperature",
        f"{temperature:.1f} °C"
    )

with col2:
    st.metric(
        "💧 Humidity",
        f"{humidity:.1f} %"
    )

with col3:
    st.metric(
        "🌬️ Pressure",
        f"{pressure:.1f} hPa"
    )

with col4:
    st.metric(
        "🎯 Trust Score",
        f"{trust_score:.1f} / 100"
    )


st.divider()


# ==========================================
# STATUS
# ==========================================

st.subheader("Sensor Health")

if status == "NORMAL":

    st.success(
        "🟢 NORMAL — Sensor reading appears trustworthy."
    )

elif status == "SUSPICIOUS":

    st.warning(
        "🟡 SUSPICIOUS — Unusual sensor pattern detected."
    )

elif status == "WARNING":

    st.warning(
        "🟠 WARNING — Sensor reading requires attention."
    )

else:

    st.error(
        "🔴 CRITICAL — Strong anomaly detected."
    )


st.info(
    f"**AI Explanation:** {explanation}"
)


st.divider()


# ==========================================
# ANOMALY SUMMARY
# ==========================================

st.subheader("📊 Anomaly Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Readings",
        len(df)
    )

with col2:
    st.metric(
        "AI Detected Anomalies",
        int(df["ai_anomaly"].sum())
    )

with col3:
    anomaly_percentage = (
        df["ai_anomaly"].mean() * 100
    )

    st.metric(
        "Anomaly Rate",
        f"{anomaly_percentage:.1f}%"
    )


# ==========================================
# SENSOR TREND
# ==========================================

st.subheader("📈 Sensor Temperature Trend")

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    df["timestamp"],
    df["temperature_c"]
)

ax.set_xlabel("Time")
ax.set_ylabel("Temperature (°C)")

plt.xticks(rotation=45)

st.pyplot(fig)


# ==========================================
# ANOMALY READINGS
# ==========================================

st.subheader("🚨 Detected Anomalies")

anomalies = df[df["ai_anomaly"] == 1].copy()

display_columns = [
    "timestamp",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "trust_score",
    "status",
    "explanation"
]

st.dataframe(
    anomalies[display_columns].tail(20),
    use_container_width=True
)


# ==========================================
# SENSOR DATA
# ==========================================

st.subheader("📋 Recent Sensor Readings")

recent = df[
    [
        "timestamp",
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "trust_score",
        "status"
    ]
].tail(20)

st.dataframe(
    recent,
    use_container_width=True
)
# ==========================================
# ANOMALY TYPE SUMMARY
# ==========================================

st.subheader("📊 Detected Anomaly Types")

detected_anomalies = df[df["ai_anomaly"] == 1]

if len(detected_anomalies) > 0:

    anomaly_counts = (
        detected_anomalies["anomaly_type"]
        .value_counts()
        .reset_index()
    )

    anomaly_counts.columns = ["Anomaly Type", "Count"]

    st.bar_chart(
        anomaly_counts.set_index("Anomaly Type")
    )

else:
    st.success("No anomalies detected.")
# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.subheader("🤖 AI Model Performance")

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

precision = precision_score(
    df["is_anomaly"],
    df["ai_anomaly"],
    zero_division=0
)

recall = recall_score(
    df["is_anomaly"],
    df["ai_anomaly"],
    zero_division=0
)

f1 = f1_score(
    df["is_anomaly"],
    df["ai_anomaly"],
    zero_division=0
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Precision",
        f"{precision * 100:.1f}%"
    )

with col2:
    st.metric(
        "Recall",
        f"{recall * 100:.1f}%"
    )

with col3:
    st.metric(
        "F1 Score",
        f"{f1 * 100:.1f}%"
    )

st.caption(
    "Evaluation is based on the synthetic sensor-fault dataset "
    "created from the NOAA baseline data."
)

st.write("### Confusion Matrix")

cm = confusion_matrix(
    df["is_anomaly"],
    df["ai_anomaly"]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Anomaly"],
    columns=["Predicted Normal", "Predicted Anomaly"]
)

st.dataframe(
    cm_df,
    use_container_width=True
)
# ==========================================
# AI DECISION PANEL
# ==========================================

st.subheader("🧠 AI Decision")

decision_col1, decision_col2 = st.columns(2)

with decision_col1:
    st.metric(
        "Sensor Trust Score",
        f"{trust_score:.1f}/100"
    )

with decision_col2:
    if status == "NORMAL":
        st.success("✅ Reading can be trusted")
    elif status == "SUSPICIOUS":
        st.warning("⚠️ Reading needs attention")
    elif status == "WARNING":
        st.warning("🚨 Possible sensor fault")
    else:
        st.error("❌ Critical anomaly detected")

st.info(
    f"**AI Explanation:** {explanation}"
)
# ==========================================
# TRUST SCORE VISUALIZATION
# ==========================================

st.subheader("🎯 Sensor Reliability")

trust_progress = max(0, min(100, int(trust_score)))

st.progress(
    trust_progress,
    text=f"Sensor Trust: {trust_score:.1f}%"
)

if trust_score >= 70:
    st.success("High confidence — sensor reading appears reliable.")
elif trust_score >= 40:
    st.warning("Moderate confidence — sensor reading should be reviewed.")
else:
    st.error("Low confidence — sensor reading may be faulty.")
# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "SkyGuard AI | AI/ML-Based Intelligent Anomaly Detection "
    "for Automatic Weather Stations"
)