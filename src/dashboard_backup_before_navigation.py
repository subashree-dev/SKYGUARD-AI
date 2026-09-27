import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛰️",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "synthetic" / "skyguard_results.csv"

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

critical_count = int((df["status"] == "CRITICAL").sum())
warning_count = int((df["status"] == "WARNING").sum())
suspicious_count = int((df["status"] == "SUSPICIOUS").sum())

average_trust = df["trust_score"].mean()

latest = df.iloc[-1]

latest_temperature = latest["temperature_c"]
latest_humidity = latest["humidity_pct"]
latest_pressure = latest["pressure_hpa"]
latest_trust = latest["trust_score"]
latest_status = latest["status"]
latest_explanation = latest["explanation"]


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.title("🛰️ SkyGuard AI")
    st.caption("AI Sensor Intelligence")

    st.divider()

    st.caption("NAVIGATION")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📡 Sensor Monitoring",
            "🚨 Anomaly Detection",
            "📊 Analytics",
            "🤖 AI Model",
            "ℹ️ System"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("SYSTEM STATUS")

    st.success("🟢 Monitoring Active")

    st.divider()

    st.caption("ABOUT")

    st.write(
        "**SkyGuard AI**\n\n"
        "Intelligent anomaly detection for "
        "Automatic Weather Stations."
    )

    st.caption("SIH26073 • AI/ML Sensor Monitoring")
# ============================================================
# HEADER
# ============================================================

st.title("🛰️ SkyGuard AI")

st.subheader(
    "Intelligent Anomaly Detection for Automatic Weather Stations"
)

st.write(
    "SkyGuard AI analyzes temperature, humidity and pressure readings "
    "to identify unusual sensor behavior and estimate sensor reliability."
)

st.divider()


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("🏠 SkyGuard AI Overview")

    st.caption(
        "Real-time sensor intelligence for Automatic Weather Stations"
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

elif page == "📡 Sensor Monitoring":

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

elif page == "🚨 Anomaly Detection":

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

elif page == "📊 Analytics":

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

elif page == "🤖 AI Model":

    st.header("🤖 AI Model Intelligence")

    st.caption(
        "Isolation Forest based unsupervised anomaly detection "
        "for Automatic Weather Station sensor readings."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL OVERVIEW
    # --------------------------------------------------------

    st.subheader("🧠 Model Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "### 🌲 Algorithm\n\n"
            "**Isolation Forest**\n\n"
            "An unsupervised machine-learning algorithm "
            "designed to identify unusual observations."
        )

    with col2:
        st.info(
            "### 📡 Input Features\n\n"
            "**Temperature**\n\n"
            "**Relative Humidity**\n\n"
            "**Pressure**"
        )

    with col3:
        st.info(
            "### 🎯 Output\n\n"
            "**Anomaly Decision**\n\n"
            "**Anomaly Score**\n\n"
            "**Trust Score**"
        )

    st.divider()

    # --------------------------------------------------------
    # HOW ISOLATION FOREST WORKS
    # --------------------------------------------------------

    st.subheader("🌲 How Isolation Forest Works")

    st.write(
        "Instead of learning every possible type of faulty sensor, "
        "Isolation Forest learns the general pattern of normal "
        "sensor observations."
    )

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.info(
            "### 1️⃣ Learn\n\n"
            "The model analyzes combinations of temperature, "
            "humidity and pressure."
        )

    with step2:
        st.info(
            "### 2️⃣ Isolate\n\n"
            "Unusual observations are easier to isolate "
            "than normal observations."
        )

    with step3:
        st.warning(
            "### 3️⃣ Detect\n\n"
            "Readings that are isolated quickly are treated "
            "as potential anomalies."
        )

    with step4:
        st.success(
            "### 4️⃣ Trust\n\n"
            "The model output is converted into a "
            "0–100 trust score."
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    st.subheader("⚙️ Model Configuration")

    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:

        st.metric(
            "Estimators",
            "200"
        )

    with config_col2:

        st.metric(
            "Contamination",
            "7%"
        )

    with config_col3:

        st.metric(
            "Random State",
            "42"
        )

    st.caption(
        "The current prototype uses 200 Isolation Forest estimators "
        "with a contamination parameter of 0.07."
    )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.subheader("📊 Prototype Model Performance")

    accuracy = 92
    precision = 36
    recall = 40
    f1 = 38

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy}%"
        )

    with col2:
        st.metric(
            "Anomaly Precision",
            f"{precision}%"
        )

    with col3:
        st.metric(
            "Anomaly Recall",
            f"{recall}%"
        )

    with col4:
        st.metric(
            "Anomaly F1",
            f"{f1}%"
        )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("🎯 Confusion Matrix")

    st.write(
        "The current prototype evaluation produced the following "
        "classification results:"
    )

    cm = pd.DataFrame(
        [
            [1063, 54],
            [45, 30]
        ],
        columns=[
            "Predicted Normal",
            "Predicted Anomaly"
        ],
        index=[
            "Actual Normal",
            "Actual Anomaly"
        ]
    )

    st.dataframe(
        cm,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE INTERPRETATION
    # --------------------------------------------------------

    st.subheader("🔍 Performance Interpretation")

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            "### 🟢 Normal Reading Detection\n\n"
            "The prototype correctly identified a large majority "
            "of normal observations."
        )

    with col2:

        st.warning(
            "### 🟡 Anomaly Detection\n\n"
            "The anomaly-class metrics show that synthetic faults "
            "are harder to identify reliably. This highlights the "
            "need for more representative labeled fault data "
            "during future validation."
        )

    st.divider()

    # --------------------------------------------------------
    # ANOMALY SCORE
    # --------------------------------------------------------

    st.subheader("📈 Anomaly Score")

    st.write(
        "Isolation Forest produces a decision-function score for "
        "each observation. SkyGuard AI uses this score as the "
        "basis for its trust calculation."
    )

    score_data = df[
        ["timestamp", "anomaly_score"]
    ].set_index("timestamp")

    st.line_chart(
        score_data,
        use_container_width=True
    )

    st.caption(
        "Lower decision-function values generally indicate "
        "greater deviation from the learned normal pattern."
    )

    st.divider()

    # --------------------------------------------------------
    # TRUST SCORE
    # --------------------------------------------------------

    st.subheader("🛡️ Trust Score Generation")

    st.write(
        "The Isolation Forest decision-function values are "
        "normalized to a 0–100 scale."
    )

    st.progress(
        min(max(float(average_trust) / 100, 0.0), 1.0)
    )

    st.metric(
        "Current Average Trust",
        f"{average_trust:.1f}/100"
    )

    st.divider()

    # --------------------------------------------------------
    # IMPORTANT EVALUATION NOTE
    # --------------------------------------------------------

    st.subheader("⚠️ Evaluation Note")

    st.warning(
        "The current performance evaluation uses NOAA/NCEI baseline "
        "weather observations with synthetically injected sensor "
        "faults. The dataset does not contain real-world ground-truth "
        "sensor-fault labels. Therefore, these metrics represent "
        "prototype testing and should not be presented as validated "
        "real-world fault-detection performance."
    )

    st.divider()

    # --------------------------------------------------------
    # AI DECISION
    # --------------------------------------------------------

    st.subheader("🤖 AI Decision Pipeline")

    st.info(
        "Sensor Reading → Feature Analysis → Isolation Forest → "
        "Anomaly Score → Anomaly Decision → Trust Score → "
        "Human Investigation"
    )

    st.caption(
        "SkyGuard AI is designed to support sensor monitoring "
        "and human investigation rather than automatically "
        "declaring the physical sensor faulty."
    )
# ============================================================
# SYSTEM
# ============================================================

elif page == "ℹ️ System":

    st.header("ℹ️ SkyGuard AI System")

    st.caption(
        "Project information, architecture, workflow and technology stack."
    )

    st.divider()

    # --------------------------------------------------------
    # PROJECT OBJECTIVE
    # --------------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.info(
        "SkyGuard AI is an intelligent anomaly-detection system "
        "for Automatic Weather Station sensors."
    )

    st.write(
        "The system monitors temperature, relative humidity and "
        "station-level pressure to identify readings that differ "
        "significantly from the learned normal sensor pattern."
    )

    st.write(
        "The primary question addressed by the system is:"
    )

    st.success(
        "💡 **Can I trust this sensor reading?**"
    )

    st.divider()

    # --------------------------------------------------------
    # PROBLEM
    # --------------------------------------------------------

    st.subheader("⚠️ Problem Being Addressed")

    st.write(
        "Automatic Weather Stations continuously generate sensor "
        "measurements. Faulty sensors can produce abnormal, "
        "inconsistent or misleading readings."
    )

    st.write(
        "SkyGuard AI provides an automated screening mechanism "
        "to identify potentially unreliable observations before "
        "they are used for further analysis."
    )

    st.divider()

    # --------------------------------------------------------
    # MONITORED PARAMETERS
    # --------------------------------------------------------

    st.subheader("📡 Monitored Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "### 🌡️ Temperature\n\n"
            "Unit: **°C**\n\n"
            "Detects unusual temperature behavior."
        )

    with col2:

        st.info(
            "### 💧 Relative Humidity\n\n"
            "Unit: **%**\n\n"
            "Detects unusual humidity readings."
        )

    with col3:

        st.info(
            "### 🌍 Station Pressure\n\n"
            "Unit: **hPa**\n\n"
            "Detects unusual pressure observations."
        )

    st.divider()

    # --------------------------------------------------------
    # SYSTEM ARCHITECTURE
    # --------------------------------------------------------

    st.subheader("🏗️ System Architecture")

    st.info(
        "NOAA/NCEI Weather Data"
    )

    st.write("⬇️")

    st.info(
        "Python + Pandas Data Processing"
    )

    st.write("⬇️")

    st.info(
        "Synthetic Sensor Fault Injection"
    )

    st.write("⬇️")

    st.info(
        "Isolation Forest Machine Learning Model"
    )

    st.write("⬇️")

    st.info(
        "Anomaly Score + Trust Score"
    )

    st.write("⬇️")

    st.success(
        "🛰️ Streamlit Interactive Dashboard"
    )

    st.divider()

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.subheader("🔄 End-to-End Workflow")

    step1, step2, step3, step4 = st.columns(4)

    with step1:

        st.info(
            "### 1️⃣ Data\n\n"
            "Weather observations are collected from "
            "the NOAA/NCEI GHCNh dataset."
        )

    with step2:

        st.info(
            "### 2️⃣ Processing\n\n"
            "Relevant temperature, humidity and pressure "
            "features are cleaned and prepared."
        )

    with step3:

        st.warning(
            "### 3️⃣ AI Detection\n\n"
            "Isolation Forest identifies observations "
            "that differ from the learned normal pattern."
        )

    with step4:

        st.success(
            "### 4️⃣ Decision Support\n\n"
            "The dashboard displays anomaly status, "
            "trust score and an explanation."
        )

    st.divider()

    # --------------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------------

    st.subheader("🛠️ Technology Stack")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### 💻 Development")

        st.write(
            """
            **Programming Language**
            - Python

            **Frontend / Dashboard**
            - Streamlit

            **Data Processing**
            - Pandas
            - NumPy

            **Visualization**
            - Streamlit Charts
            - Matplotlib
            """
        )

    with col2:

        st.write("### 🤖 AI & Infrastructure")

        st.write(
            """
            **Machine Learning**
            - Scikit-learn
            - Isolation Forest

            **Dataset**
            - NOAA/NCEI GHCNh

            **Version Control**
            - Git
            - GitHub

            **Deployment**
            - Streamlit Community Cloud
            """
        )

    st.divider()

    # --------------------------------------------------------
    # PROJECT DATA
    # --------------------------------------------------------

    st.subheader("📊 Current Prototype Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Baseline Readings",
            "1,192"
        )

    with col2:
        st.metric(
            "Synthetic Anomalies",
            "75"
        )

    with col3:
        st.metric(
            "Sensor Features",
            "3"
        )

    with col4:
        st.metric(
            "ML Algorithm",
            "Isolation Forest"
        )

    st.divider()

    # --------------------------------------------------------
    # FAULT TYPES
    # --------------------------------------------------------

    st.subheader("🧩 Prototype Fault Scenarios")

    fault_col1, fault_col2 = st.columns(2)

    with fault_col1:

        st.write(
            """
            **Temperature Faults**
            - Temperature spike
            - Temperature drift
            - Temperature stuck
            """
        )

    with fault_col2:

        st.write(
            """
            **Humidity Faults**
            - Humidity spike

            Additional fault scenarios can be incorporated "
            "when representative labeled sensor data becomes available."
            """
        )

    st.divider()

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.subheader("⚠️ Current Prototype Limitations")

    st.warning(
        "The current prototype uses NOAA/NCEI baseline observations "
        "and synthetically injected sensor faults because the "
        "baseline dataset does not provide ground-truth labels "
        "for real sensor failures."
    )

    st.warning(
        "Therefore, the current performance metrics demonstrate "
        "prototype behavior and should not be interpreted as "
        "validated real-world sensor-fault detection performance."
    )

    st.info(
        "🚀 Future development can include larger labeled fault "
        "datasets, additional sensor consistency checks, "
        "historical monitoring and improved model validation."
    )

    st.divider()

    # --------------------------------------------------------
    # FINAL PROJECT MESSAGE
    # --------------------------------------------------------

    st.success(
        "🛰️ **SkyGuard AI** — Intelligent Sensor Anomaly Detection"
    )

    st.caption(
        "SIH26073 • Automatic Weather Station Sensor Intelligence"
    )