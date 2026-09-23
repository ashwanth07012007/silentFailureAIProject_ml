import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/sensor_data.csv")


# ==========================================
# 2. NORMAL BASELINE
# ==========================================

normal = df[df["timestamp"] < 700]

normal_values = {
    "temperature": normal["temperature"].mean(),
    "current": normal["current"].mean(),
    "vibration": normal["vibration"].mean(),
    "pressure": normal["pressure"].mean()
}


# ==========================================
# 3. SENSOR SCORE FUNCTION
# ==========================================

def sensor_score(value, normal_value, limit):

    deviation = abs(value - normal_value) / normal_value

    score = deviation / limit

    return min(score, 1)


# ==========================================
# 4. CALCULATE RISK FOR EVERY TIMESTAMP
# ==========================================

def calculate_risk(row):

    temperature_score = sensor_score(
        row["temperature"],
        normal_values["temperature"],
        0.40
    )

    current_score = sensor_score(
        row["current"],
        normal_values["current"],
        0.40
    )

    vibration_score = sensor_score(
        row["vibration"],
        normal_values["vibration"],
        1.00
    )

    pressure_score = sensor_score(
        row["pressure"],
        normal_values["pressure"],
        0.50
    )

    risk = (
        temperature_score * 25 +
        current_score * 25 +
        vibration_score * 25 +
        pressure_score * 25
    )

    return min(risk, 100)


df["risk"] = df.apply(calculate_risk, axis=1)


# ==========================================
# 5. SMOOTH RISK
# ==========================================

df["rolling_risk"] = (
    df["risk"]
    .rolling(30)
    .mean()
)


# ==========================================
# 6. CREATE GRAPH
# ==========================================

plt.figure(figsize=(14, 7))

plt.plot(
    df["timestamp"],
    df["rolling_risk"],
    linewidth=2,
    label="Machine Risk"
)


# ==========================================
# 7. RISK THRESHOLD LINES
# ==========================================

plt.axhline(
    y=40,
    linestyle="--",
    label="Medium Risk"
)

plt.axhline(
    y=70,
    linestyle="--",
    label="High Risk"
)


# ==========================================
# 8. IMPORTANT MACHINE STAGES
# ==========================================

plt.axvline(
    x=700,
    linestyle="--",
    label="Degradation Begins"
)

plt.axvline(
    x=900,
    linestyle="--",
    label="Severe Failure"
)


# ==========================================
# 9. LABELS
# ==========================================

plt.title(
    "Silent Failure Detector - Machine Risk Over Time"
)

plt.xlabel("Time")

plt.ylabel("Failure Risk (0-100)")

plt.ylim(0, 105)

plt.legend()

plt.grid(True)


# ==========================================
# 10. DISPLAY
# ==========================================

plt.show()