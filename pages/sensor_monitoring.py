import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.shared import *


st.header("📡 Sensor Monitoring")

st.caption(
    "Inspect weather-station measurements and evaluate sensor reliability."
)

st.divider()

# --------------------------------------------------------
# SENSOR SELECTION
# --------------------------------------------------------

st.subheader("🎚️ Select Sensor Reading")

selected_index = st.slider(
    "Reading Index",
    min_value=0,
    max_value=len(df) - 1,
    value=len(df) - 1,
    key="monitoring_slider"
)

selected = df.iloc[selected_index]

st.caption(
    f"Reading timestamp: {selected['timestamp']}"
)

st.divider()

# --------------------------------------------------------
# SENSOR VALUES
# --------------------------------------------------------

st.subheader("🌡️ Current Sensor Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🌡️ Temperature",
        f"{selected['temperature_c']:.2f} °C"
    )

with col2:
    st.metric(
        "💧 Relative Humidity",
        f"{selected['humidity_pct']:.2f} %"
    )

with col3:
    st.metric(
        "🌍 Pressure",
        f"{selected['pressure_hpa']:.2f} hPa"
    )

st.divider()

# --------------------------------------------------------
# SENSOR HEALTH
# --------------------------------------------------------

st.subheader("🩺 Sensor Health")

health_col, status_col = st.columns(2)

with health_col:

    trust = float(selected["trust_score"])

    st.metric(
        "🛡️ Sensor Trust Score",
        f"{trust:.1f}/100"
    )

    st.progress(
        min(max(trust / 100, 0.0), 1.0)
    )

    if trust >= 70:
        st.success(
            "High confidence in this sensor reading."
        )

    elif trust >= 40:
        st.warning(
            "Moderate confidence — reading should be monitored."
        )

    else:
        st.error(
            "Low confidence — reading requires investigation."
        )

with status_col:

    status = selected["status"]

    if status == "NORMAL":

        st.success(
            "🟢 SENSOR STATUS: NORMAL"
        )

        st.write(
            "The AI model considers this reading "
            "consistent with the learned normal pattern."
        )

    elif status == "SUSPICIOUS":

        st.warning(
            "🟡 SENSOR STATUS: SUSPICIOUS"
        )

        st.write(
            "The reading differs from the learned normal pattern "
            "and should be monitored."
        )

    elif status == "WARNING":

        st.warning(
            "🟠 SENSOR STATUS: WARNING"
        )

        st.write(
            "The reading shows unusual behavior "
            "and may require investigation."
        )

    else:

        st.error(
            "🔴 SENSOR STATUS: CRITICAL"
        )

        st.write(
            "The AI model has identified a highly unusual "
            "sensor reading."
        )

st.divider()

# --------------------------------------------------------
# AI EXPLANATION
# --------------------------------------------------------

st.subheader("🤖 AI Explanation")

st.info(
    selected["explanation"]
)

st.divider()

# --------------------------------------------------------
# SENSOR READING DETAILS
# --------------------------------------------------------

st.subheader("📋 Reading Details")

details_col1, details_col2 = st.columns(2)

with details_col1:

    st.write(
        f"**Station ID:** {selected['station_id']}"
    )

    st.write(
        f"**Timestamp:** {selected['timestamp']}"
    )

    st.write(
        f"**Reading Index:** {selected_index}"
    )

with details_col2:

    st.write(
        f"**AI Prediction:** "
        f"{'Anomaly' if selected['ai_anomaly'] == 1 else 'Normal'}"
    )

    st.write(
        f"**Anomaly Score:** "
        f"{selected['anomaly_score']:.4f}"
    )

    st.write(
        f"**Trust Score:** "
        f"{selected['trust_score']:.1f}/100"
    )

st.divider()

# --------------------------------------------------------
# RECENT SENSOR READINGS
# --------------------------------------------------------

st.subheader("📈 Recent Sensor Readings")

recent = df[
    [
        "timestamp",
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "trust_score",
        "status"
    ]
].tail(15).copy()

recent["timestamp"] = recent["timestamp"].dt.strftime(
    "%Y-%m-%d %H:%M"
)

st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True
)

st.divider()

# --------------------------------------------------------
# TEMPERATURE TREND
# --------------------------------------------------------

st.subheader("🌡️ Temperature Trend")

temperature_data = df[
    ["timestamp", "temperature_c"]
].set_index("timestamp")

st.line_chart(
    temperature_data,
    use_container_width=True
)

st.caption(
    "Temperature trend across the available weather-station readings."
)
# ============================================================
# ANOMALY DETECTION
# ============================================================

