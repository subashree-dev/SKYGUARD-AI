import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.shared import *


st.header("📊 Sensor Analytics")

st.caption(
    "Visual analysis of weather-station readings, "
    "sensor reliability and anomaly patterns."
)

st.divider()

# --------------------------------------------------------
# OVERVIEW METRICS
# --------------------------------------------------------

st.subheader("📌 Analytics Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Readings",
        f"{total_readings:,}"
    )

with col2:
    st.metric(
        "Normal",
        f"{normal_readings:,}"
    )

with col3:
    st.metric(
        "AI Anomalies",
        f"{ai_anomalies:,}"
    )

with col4:
    st.metric(
        "Average Trust",
        f"{average_trust:.1f}/100"
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
    "Temperature variation across the available "
    "weather-station observations."
)

st.divider()

# --------------------------------------------------------
# HUMIDITY TREND
# --------------------------------------------------------

st.subheader("💧 Humidity Trend")

humidity_data = df[
    ["timestamp", "humidity_pct"]
].set_index("timestamp")

st.line_chart(
    humidity_data,
    use_container_width=True
)

st.caption(
    "Relative humidity variation across the observations."
)

st.divider()

# --------------------------------------------------------
# PRESSURE TREND
# --------------------------------------------------------

st.subheader("🌍 Pressure Trend")

pressure_data = df[
    ["timestamp", "pressure_hpa"]
].set_index("timestamp")

st.line_chart(
    pressure_data,
    use_container_width=True
)

st.caption(
    "Station-level pressure variation across the dataset."
)

st.divider()

# --------------------------------------------------------
# ANOMALY ANALYTICS
# --------------------------------------------------------

st.subheader("🚨 Anomaly Analytics")

anomaly_col1, anomaly_col2 = st.columns(2)

with anomaly_col1:

    st.write("### Anomaly Status")

    status_counts = df["status"].value_counts()

    st.bar_chart(
        status_counts,
        use_container_width=True
    )

with anomaly_col2:

    st.write("### Prototype Fault Types")

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

    else:

        st.info(
            "No prototype fault categories available."
        )

st.divider()

# --------------------------------------------------------
# TRUST SCORE ANALYSIS
# --------------------------------------------------------

st.subheader("🛡️ Sensor Trust Analysis")

trust_data = df[
    ["timestamp", "trust_score"]
].set_index("timestamp")

st.line_chart(
    trust_data,
    use_container_width=True
)

st.caption(
    "Trust score is derived from the Isolation Forest "
    "decision function and normalized to a 0–100 scale."
)

st.divider()

# --------------------------------------------------------
# TRUST CATEGORIES
# --------------------------------------------------------

st.subheader("📊 Trust Score Categories")

high_trust = int(
    (df["trust_score"] >= 70).sum()
)

medium_trust = int(
    (
        (df["trust_score"] >= 40)
        & (df["trust_score"] < 70)
    ).sum()
)

low_trust = int(
    (df["trust_score"] < 40).sum()
)

col1, col2, col3 = st.columns(3)

with col1:

    st.success(
        "🟢 High Trust"
    )

    st.metric(
        "Readings ≥ 70",
        f"{high_trust:,}"
    )

with col2:

    st.warning(
        "🟡 Medium Trust"
    )

    st.metric(
        "Readings 40–69",
        f"{medium_trust:,}"
    )

with col3:

    st.error(
        "🔴 Low Trust"
    )

    st.metric(
        "Readings < 40",
        f"{low_trust:,}"
    )

st.divider()

# --------------------------------------------------------
# RECENT DATA
# --------------------------------------------------------

st.subheader("📋 Recent Sensor Data")

recent_data = df[
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
    recent_data,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.info(
    "💡 **Analytics Note:** The current dataset is based on "
    "NOAA/NCEI baseline weather observations with synthetically "
    "injected sensor faults for prototype evaluation."
)

# ============================================================
# AI MODEL
# ============================================================

