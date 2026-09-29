import pandas as pd


def detect_sensor_rules(df):

    df = df.copy()

    # Make sure data is in chronological order
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # TEMPERATURE STUCK DETECTION
    # --------------------------------------------------------

    same_temperature = (
        df["temperature_c"]
        .eq(df["temperature_c"].shift())
    )

    temperature_stuck_group = (
        same_temperature
        .groupby(
            (~same_temperature).cumsum()
        )
        .transform("sum")
    )

    df["temperature_stuck"] = (
        temperature_stuck_group >= 5
    )

    # --------------------------------------------------------
    # HUMIDITY STUCK DETECTION
    # --------------------------------------------------------

    same_humidity = (
        df["humidity_pct"]
        .eq(df["humidity_pct"].shift())
    )

    humidity_stuck_group = (
        same_humidity
        .groupby(
            (~same_humidity).cumsum()
        )
        .transform("sum")
    )

    df["humidity_stuck"] = (
        humidity_stuck_group >= 5
    )

    # --------------------------------------------------------
    # PRESSURE STUCK DETECTION
    # --------------------------------------------------------

    same_pressure = (
        df["pressure_hpa"]
        .eq(df["pressure_hpa"].shift())
    )

    pressure_stuck_group = (
        same_pressure
        .groupby(
            (~same_pressure).cumsum()
        )
        .transform("sum")
    )

    df["pressure_stuck"] = (
        pressure_stuck_group >= 5
    )

    # --------------------------------------------------------
    # FINAL SENSOR RULE
    # --------------------------------------------------------

    df["rule_anomaly"] = (
        df["temperature_stuck"]
        | df["humidity_stuck"]
        | df["pressure_stuck"]
    )

    return df