import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛰️",
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

st.title("🛰️ SkyGuard AI")

st.subheader(
    "Intelligent Anomaly Detection for Automatic Weather Stations"
)

st.write(
    "Monitor sensor reliability, detect abnormal readings, "
    "and identify potential weather-station sensor faults using AI."
)

st.caption(
    "AI-powered monitoring • Temperature • Humidity • Pressure"
)

st.divider()

# ==========================================
# SYSTEM INFORMATION
# ==========================================

info1, info2, info3 = st.columns(3)

with info1:
    st.info(
        "📡 **WEATHER STATION**\n\n"
        "NUMGAMBAKKAM\n\n"
        "NOAA GHCNh baseline data"
    )

with info2:
    st.info(
        "🤖 **AI ENGINE**\n\n"
        "Isolation Forest\n\n"
        "Unsupervised anomaly detection"
    )

with info3:
    st.info(
        "🛡️ **SYSTEM PURPOSE**\n\n"
        "Sensor Anomaly Detection\n\n"
        "Trust-score based monitoring"
    )
st.divider()


# ==========================================
# SENSOR READING ANALYSIS
# ==========================================

st.subheader("🔎 Sensor Reading Analysis")

st.write(
    "Select a sensor reading to inspect its measurements "
    "and AI-generated reliability assessment."
)

anomaly_indices = df.index[
    df["ai_anomaly"] == 1
].tolist()

col_select, col_button = st.columns([4, 1])

with col_select:
    selected_index = st.slider(
        "Sensor reading index",
        min_value=0,
        max_value=len(df) - 1,
        value=len(df) - 1
    )

with col_button:
    st.write("")
    st.write("")
    if st.button(
        "🚨 Find Anomaly",
        use_container_width=True
    ):
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

st.caption(
    f"📅 Selected reading: {timestamp}  |  "
    f"Reading #{selected_index + 1} of {len(df)}"
)
# ==========================================
# CURRENT SENSOR READING
# ==========================================

st.subheader("📡 Current Sensor Reading")

st.caption(
    f"📅 Reading timestamp: {timestamp}"
)

st.write("### Live Sensor Values")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🌡️ Temperature",
        f"{temperature:.1f} °C",
        help="Current temperature measured by the weather station."
    )

with col2:
    st.metric(
        "💧 Humidity",
        f"{humidity:.1f} %",
        help="Current relative humidity measured by the sensor."
    )

with col3:
    st.metric(
        "🌬️ Pressure",
        f"{pressure:.1f} hPa",
        help="Current station-level atmospheric pressure."
    )

with col4:
    st.metric(
        "🎯 Trust Score",
        f"{trust_score:.1f} / 100",
        help="AI-derived reliability score for the selected reading."
    )


st.divider()


# ==========================================
# SENSOR HEALTH
# ==========================================

st.subheader("🩺 Sensor Health")

health_col1, health_col2 = st.columns([1, 2])

with health_col1:

    if status == "NORMAL":
        st.success(
            "🟢 NORMAL\n\n"
            "Reading appears trustworthy."
        )

    elif status == "SUSPICIOUS":
        st.warning(
            "🟡 SUSPICIOUS\n\n"
            "Unusual sensor pattern detected."
        )

    elif status == "WARNING":
        st.warning(
            "🟠 WARNING\n\n"
            "Reading requires attention."
        )

    else:
        st.error(
            "🔴 CRITICAL\n\n"
            "Strong anomaly detected."
        )

with health_col2:

    st.write("### 🧠 AI Analysis")

    st.info(
        explanation
    )

    st.caption(
        "The AI evaluates the combined temperature, humidity "
        "and pressure pattern against the learned normal behaviour."
    )

st.divider()


# ==========================================
# SENSOR RELIABILITY
# ==========================================

st.subheader("🎯 Sensor Reliability")

reliability_col1, reliability_col2 = st.columns([1, 2])

with reliability_col1:
    st.metric(
        "Sensor Trust",
        f"{trust_score:.1f} / 100"
    )

with reliability_col2:
    trust_progress = max(
        0,
        min(100, int(trust_score))
    )

    st.progress(
        trust_progress,
        text=f"Reliability Level: {trust_score:.1f}%"
    )

    if trust_score >= 70:
        st.success(
            "🟢 High reliability — the reading closely "
            "matches the learned normal sensor pattern."
        )

    elif trust_score >= 40:
        st.warning(
            "🟡 Moderate reliability — the reading should "
            "be reviewed before being treated as fully reliable."
        )

    else:
        st.error(
            "🔴 Low reliability — the reading significantly "
            "differs from the learned normal pattern."
        )

st.divider()


# ==========================================
# ANOMALY SUMMARY
# ==========================================

st.subheader("📊 Anomaly Summary")

total_readings = len(df)
detected_anomalies = int(df["ai_anomaly"].sum())
anomaly_percentage = (
    detected_anomalies / total_readings * 100
)

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric(
        "📡 Total Readings",
        f"{total_readings:,}"
    )

with summary_col2:
    st.metric(
        "🚨 AI Detected",
        f"{detected_anomalies:,}"
    )

with summary_col3:
    st.metric(
        "📈 Anomaly Rate",
        f"{anomaly_percentage:.1f}%"
    )

st.caption(
    "Summary of readings classified by the Isolation Forest "
    "anomaly-detection model."
)
st.divider()


# ==========================================
# SENSOR TEMPERATURE TREND
# ==========================================
st.subheader("📈 Sensor Temperature Trend")

st.caption(
    "Temperature variation across the monitored period. "
    "Unusual readings are identified by the AI anomaly detector."
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    df["timestamp"],
    df["temperature_c"],
    linewidth=1.5
)

ax.set_xlabel("Time")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Temperature Sensor Behaviour")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)
st.divider()


# ==========================================
# DETECTED ANOMALIES
# ==========================================

st.subheader("🚨 Detected Anomalies")

st.caption(
    "Sensor readings flagged by the Isolation Forest model "
    "as significantly different from the learned normal pattern."
)

anomalies = df[
    df["ai_anomaly"] == 1
].copy()

display_columns = [
    "timestamp",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "trust_score",
    "status",
    "explanation"
]

if len(anomalies) > 0:

    st.metric(
        "Total AI-Flagged Readings",
        len(anomalies)
    )

    st.dataframe(
        anomalies[
            display_columns
        ].tail(20),
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "✅ No anomalous sensor readings detected."
    )
st.divider()


# ==========================================
# ANOMALY TYPE SUMMARY
# ==========================================

st.subheader("📊 Detected Anomaly Types")

st.caption(
    "Breakdown of the different fault patterns identified "
    "within the sensor dataset."
)

detected_anomalies = df[
    df["ai_anomaly"] == 1
]

if len(detected_anomalies) > 0:

    anomaly_counts = (
        detected_anomalies["anomaly_type"]
        .value_counts()
        .reset_index()
    )

    anomaly_counts.columns = [
        "Anomaly Type",
        "Count"
    ]

    st.bar_chart(
        anomaly_counts.set_index("Anomaly Type")
    )

else:

    st.success(
        "✅ No anomalies detected."
    )
st.divider()


# ==========================================
# RECENT SENSOR READINGS
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


st.divider()


# ==========================================
# AI MODEL PERFORMANCE
# ==========================================

st.subheader("🤖 AI Model Performance")

st.caption(
    "Prototype evaluation of the Isolation Forest anomaly detector "
    "using synthetic sensor faults generated from the NOAA baseline dataset."
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

performance_col1, performance_col2, performance_col3 = st.columns(3)

with performance_col1:
    st.metric(
        "🎯 Precision",
        f"{precision * 100:.1f}%"
    )

with performance_col2:
    st.metric(
        "🔎 Recall",
        f"{recall * 100:.1f}%"
    )

with performance_col3:
    st.metric(
        "⚖️ F1 Score",
        f"{f1 * 100:.1f}%"
    )

st.write("### 📋 Confusion Matrix")

cm = confusion_matrix(
    df["is_anomaly"],
    df["ai_anomaly"]
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Normal",
        "Actual Anomaly"
    ],
    columns=[
        "Predicted Normal",
        "Predicted Anomaly"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True,
    hide_index=False
)

st.info(
    "ℹ️ These metrics are based on synthetic fault-injection data "
    "created from the NOAA baseline dataset. They demonstrate "
    "prototype behaviour and should not be interpreted as "
    "real-world fault-detection accuracy."
)


st.divider()


# ==========================================
# FINAL AI DECISION
# ==========================================

st.subheader("🧠 AI Decision")

st.caption(
    "Final AI assessment for the selected sensor reading."
)

decision_col1, decision_col2 = st.columns(2)

with decision_col1:

    st.metric(
        "🎯 Sensor Trust Score",
        f"{trust_score:.1f} / 100"
    )

with decision_col2:

    if status == "NORMAL":

        st.success(
            "✅ READING APPEARS RELIABLE"
        )

    elif status == "SUSPICIOUS":

        st.warning(
            "⚠️ READING NEEDS ATTENTION"
        )

    elif status == "WARNING":

        st.warning(
            "🚨 POSSIBLE SENSOR FAULT"
        )

    else:

        st.error(
            "❌ CRITICAL ANOMALY DETECTED"
        )

st.write("### 🧠 Why did the AI flag this reading?")

st.info(
    explanation
)

st.caption(
    "The decision is based on how different the combined "
    "sensor measurements are from the learned normal pattern."
)
st.divider()

st.subheader("⚙️ How SkyGuard AI Works")

workflow1, workflow2, workflow3, workflow4 = st.columns(4)

with workflow1:
    st.info(
        "📡 **1. SENSOR DATA**\n\n"
        "Temperature, humidity and pressure readings "
        "are collected from the weather-station dataset."
    )

with workflow2:
    st.info(
        "🧹 **2. DATA PROCESSING**\n\n"
        "Sensor values are cleaned and prepared "
        "for anomaly detection."
    )

with workflow3:
    st.info(
        "🤖 **3. AI DETECTION**\n\n"
        "Isolation Forest learns the normal sensor "
        "pattern and identifies unusual readings."
    )

with workflow4:
    st.info(
        "🎯 **4. TRUST SCORE**\n\n"
        "Each reading receives an AI-based reliability "
        "score and health status."
    )
st.divider()

st.write("### 🛰️ SkyGuard AI")

st.caption(
    "Intelligent Anomaly Detection for Automatic Weather Stations"
)

st.caption(
    "Temperature • Humidity • Pressure • Isolation Forest • Trust Score"
)

st.caption(
    "Prototype developed for SIH 2026 | "
    "Evaluation uses synthetic sensor-fault data generated "
    "from a NOAA baseline dataset."
)