import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.shared import *


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