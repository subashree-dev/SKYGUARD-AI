import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.shared import *


st.title("🛰️ SkyGuard AI")

st.subheader("Intelligent Sensor Monitoring & Anomaly Detection")

st.caption(
    "Real-time monitoring of temperature, humidity, and pressure "
    "for Automatic Weather Stations."
)
# --------------------------------------------------------
# TOP STATUS
# --------------------------------------------------------

if critical_count > 0:
    st.warning(
        f"⚠️ Monitoring active — {critical_count} critical "
        "sensor event(s) require attention."
    )
else:
    st.success(
        "🟢 Monitoring active — no critical sensor events detected."
    )

st.divider()

# --------------------------------------------------------
# KEY METRICS
# --------------------------------------------------------
st.subheader("📊 System Snapshot")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📡 Total Readings",
        f"{total_readings:,}"
    )

with col2:
    st.metric(
        "🚨 AI Anomalies",
        f"{ai_anomalies:,}"
    )

with col3:
    st.metric(
        "🛡️ Average Trust",
        f"{average_trust:.1f}/100"
    )

with col4:
    st.metric(
        "⚠️ Anomaly Rate",
        f"{anomaly_rate:.1f}%"
    )
st.divider()

# --------------------------------------------------------
# CURRENT SENSOR STATUS
# --------------------------------------------------------

st.subheader("📡 Latest Sensor Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🌡️ Temperature",
        f"{latest_temperature:.1f} °C"
    )

with col2:
    st.metric(
        "💧 Humidity",
        f"{latest_humidity:.1f} %"
    )

with col3:
    st.metric(
        "🌍 Pressure",
        f"{latest_pressure:.1f} hPa"
    )

with col4:
    st.metric(
        "🛡️ Trust Score",
        f"{latest_trust:.1f}/100"
    )

# Status message

if latest_status == "NORMAL":
    st.success(
        f"🟢 Current sensor status: **{latest_status}**"
    )

elif latest_status == "SUSPICIOUS":
    st.warning(
        f"🟡 Current sensor status: **{latest_status}**"
    )

elif latest_status == "WARNING":
    st.warning(
        f"🟠 Current sensor status: **{latest_status}**"
    )

else:
    st.error(
        f"🔴 Current sensor status: **{latest_status}**"
    )

st.info(
    f"🤖 **AI Explanation:** {latest_explanation}"
)

st.divider()

# --------------------------------------------------------
# ANOMALY SUMMARY
# --------------------------------------------------------

st.subheader("🚨 Anomaly Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🟢 Normal",
        f"{normal_readings:,}"
    )

with col2:
    st.metric(
        "🟡 Suspicious",
        f"{suspicious_count:,}"
    )

with col3:
    st.metric(
        "🟠 Warning",
        f"{warning_count:,}"
    )

with col4:
    st.metric(
        "🔴 Critical",
        f"{critical_count:,}"
    )

st.divider()

# --------------------------------------------------------
# QUICK ANALYTICS
# --------------------------------------------------------

st.subheader("📈 Quick Analytics")

chart_col, status_col = st.columns([2, 1])

with chart_col:

    st.write("### 🌡️ Temperature Trend")

    temperature_chart = df[
        ["timestamp", "temperature_c"]
    ].set_index("timestamp")

    st.line_chart(
        temperature_chart,
        use_container_width=True
    )

with status_col:

    st.write("### 🛡️ Sensor Reliability")

    st.metric(
        "Average Trust",
        f"{average_trust:.1f}/100"
    )

    st.progress(
        min(max(average_trust / 100, 0.0), 1.0)
    )

    st.write("")

    st.metric(
        "Normal Readings",
        f"{normal_readings:,}"
    )

    st.metric(
        "Critical Events",
        f"{critical_count:,}"
    )

st.divider()

# --------------------------------------------------------
# HOW SKYGUARD WORKS
# --------------------------------------------------------

st.subheader("🔄 How SkyGuard AI Works")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info(
        "### 1️⃣ Collect\n\n"
        "Weather-station readings are collected for "
        "temperature, humidity and pressure."
    )

with col2:
    st.info(
        "### 2️⃣ Analyze\n\n"
        "Isolation Forest learns the normal pattern "
        "of sensor measurements."
    )

with col3:
    st.warning(
        "### 3️⃣ Detect\n\n"
        "Unusual combinations of readings are flagged "
        "as potential anomalies."
    )

with col4:
    st.success(
        "### 4️⃣ Trust\n\n"
        "Every reading receives a trust score from "
        "0 to 100 for easier monitoring."
    )

st.divider()

# --------------------------------------------------------
# QUICK NAVIGATION
# --------------------------------------------------------

st.subheader("⚡ Explore SkyGuard AI")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "📡 **Sensor Monitoring**\n\n"
        "Inspect individual sensor readings, "
        "health and reliability."
    )

with col2:
    st.info(
        "🚨 **Anomaly Detection**\n\n"
        "Find unusual readings and inspect "
        "AI-generated anomaly decisions."
    )

with col3:
    st.info(
        "🤖 **AI Model**\n\n"
        "View Isolation Forest performance "
        "and prototype evaluation."
    )

st.divider()

st.caption(
    "SkyGuard AI • SIH26073 • Intelligent Anomaly Detection "
    "for Automatic Weather Stations"
)
# ============================================================
# SENSOR MONITORING
# ============================================================

