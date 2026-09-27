import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.shared import *


st.header("🚨 Anomaly Detection")

st.caption(
    "Identify unusual weather-station readings using the "
    "Isolation Forest anomaly-detection model."
)

st.divider()

# --------------------------------------------------------
# READING SELECTION
# --------------------------------------------------------

st.subheader("🔎 Analyze a Sensor Reading")

selected_index = st.slider(
    "Select Reading Index",
    min_value=0,
    max_value=len(df) - 1,
    value=len(df) - 1,
    key="anomaly_detection_slider"
)

selected = df.iloc[selected_index]

st.caption(
    f"Timestamp: {selected['timestamp']}"
)

st.divider()

# --------------------------------------------------------
# SENSOR VALUES
# --------------------------------------------------------

st.subheader("📡 Sensor Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🌡️ Temperature",
        f"{selected['temperature_c']:.2f} °C"
    )

with col2:
    st.metric(
        "💧 Humidity",
        f"{selected['humidity_pct']:.2f} %"
    )

with col3:
    st.metric(
        "🌍 Pressure",
        f"{selected['pressure_hpa']:.2f} hPa"
    )

st.divider()

# --------------------------------------------------------
# AI DECISION
# --------------------------------------------------------

st.subheader("🤖 AI Decision")

if selected["ai_anomaly"] == 1:

    st.error(
        "🚨 ANOMALY DETECTED"
    )

    st.write(
        "The Isolation Forest model classified this "
        "sensor reading as unusual compared with the "
        "learned normal pattern."
    )

else:

    st.success(
        "🟢 NORMAL READING"
    )

    st.write(
        "The Isolation Forest model found this reading "
        "consistent with the learned normal pattern."
    )

# --------------------------------------------------------
# TRUST SCORE
# --------------------------------------------------------

trust = float(selected["trust_score"])

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "🛡️ Trust Score",
        f"{trust:.1f}/100"
    )

    st.progress(
        min(max(trust / 100, 0.0), 1.0)
    )

with col2:

    st.metric(
        "📊 Anomaly Score",
        f"{selected['anomaly_score']:.4f}"
    )

st.divider()

# --------------------------------------------------------
# AI EXPLANATION
# --------------------------------------------------------

st.subheader("💡 Why was this reading flagged?")

st.info(
    selected["explanation"]
)

st.caption(
    "The explanation describes observable sensor behavior. "
    "The synthetic fault label used during prototype testing "
    "is not treated as an AI-generated explanation."
)

st.divider()

# --------------------------------------------------------
# ANOMALY STATUS
# --------------------------------------------------------

st.subheader("🚦 Anomaly Status")

status = selected["status"]

if status == "NORMAL":

    st.success(
        "🟢 NORMAL — No significant deviation detected."
    )

elif status == "SUSPICIOUS":

    st.warning(
        "🟡 SUSPICIOUS — The reading differs from the "
        "learned normal pattern."
    )

elif status == "WARNING":

    st.warning(
        "🟠 WARNING — The reading requires closer inspection."
    )

else:

    st.error(
        "🔴 CRITICAL — The reading shows a strong deviation "
        "from the learned normal pattern."
    )

st.divider()

# --------------------------------------------------------
# ANOMALY SUMMARY
# --------------------------------------------------------

st.subheader("📊 Anomaly Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Readings",
        f"{total_readings:,}"
    )

with col2:
    st.metric(
        "AI Anomalies",
        f"{ai_anomalies:,}"
    )

with col3:
    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.1f}%"
    )

with col4:
    st.metric(
        "Critical",
        f"{critical_count:,}"
    )

st.divider()

# --------------------------------------------------------
# DETECTED ANOMALIES TABLE
# --------------------------------------------------------

st.subheader("📋 Detected Anomalies")

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

st.dataframe(
    anomalies[display_columns],
    use_container_width=True,
    hide_index=True
)

st.divider()

# --------------------------------------------------------
# ANOMALY TYPES
# --------------------------------------------------------

st.subheader("🧩 Fault Types in Prototype Dataset")

anomaly_type_counts = (
    df[df["anomaly_type"] != "normal"]
    ["anomaly_type"]
    .value_counts()
)

if len(anomaly_type_counts) > 0:

    st.bar_chart(
        anomaly_type_counts,
        use_container_width=True
    )

st.caption(
    "These fault categories come from the synthetic fault-injection "
    "dataset used to evaluate the prototype."
)
# ============================================================
# ANALYTICS
# ============================================================

