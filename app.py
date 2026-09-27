import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛰️",
    layout="wide"
)


# ============================================================
# PAGE DEFINITIONS
# ============================================================

overview = st.Page(
    "pages/overview.py",
    title="Overview",
    icon="🏠"
)

sensor_monitoring = st.Page(
    "pages/sensor_monitoring.py",
    title="Sensor Monitoring",
    icon="📡"
)

anomaly_detection = st.Page(
    "pages/anomaly_detection.py",
    title="Anomaly Detection",
    icon="🚨"
)

analytics = st.Page(
    "pages/analytics.py",
    title="Analytics",
    icon="📊"
)

ai_model = st.Page(
    "pages/ai_model.py",
    title="AI Model",
    icon="🤖"
)

system = st.Page(
    "pages/system.py",
    title="System",
    icon="ℹ️"
)


# ============================================================
# NAVIGATION
# ============================================================

pg = st.navigation(
    [
        overview,
        sensor_monitoring,
        anomaly_detection,
        analytics,
        ai_model,
        system
    ],
    position="sidebar"
)


# ============================================================
# SIDEBAR BRANDING
# ============================================================

with st.sidebar:

    st.caption("AI SENSOR INTELLIGENCE")

    st.divider()

    st.caption("SYSTEM STATUS")
    st.success("🟢 Monitoring Active")

    st.divider()

    st.caption("ABOUT")

    st.write(
        "SkyGuard AI monitors temperature, humidity, "
        "and pressure readings from Automatic Weather Stations "
        "and identifies unusual sensor behavior."
    )

    st.caption("SIH26073 • AI/ML Sensor Monitoring")


# ============================================================
# RUN SELECTED PAGE
# ============================================================

pg.run()