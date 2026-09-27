import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.shared import *


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

