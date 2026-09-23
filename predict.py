import pandas as pd
from sklearn.ensemble import IsolationForest

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/sensor_data.csv")

features = [
    "temperature",
    "current",
    "vibration",
    "pressure"
]


# ==========================================
# 2. NORMAL BASELINE
# ==========================================

# First 700 records represent healthy behavior

normal = df[df["timestamp"] < 700]

normal_values = {
    "temperature": normal["temperature"].mean(),
    "current": normal["current"].mean(),
    "vibration": normal["vibration"].mean(),
    "pressure": normal["pressure"].mean()
}


# ==========================================
# 3. ANOMALY DETECTION
# ==========================================

# Train the anomaly detector only on healthy data

X_normal = normal[features]

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X_normal)

# Predict all sensor readings

df["prediction"] = model.predict(df[features])

df["anomaly"] = df["prediction"].map({
    1: False,
    -1: True
})


# ==========================================
# 4. SENSOR RISK FUNCTION
# ==========================================

def sensor_score(value, normal_value, limit):

    deviation = abs(value - normal_value) / normal_value

    score = deviation / limit

    return min(score, 1)


# ==========================================
# 5. CALCULATE RISK
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


# Calculate risk for every timestamp

df["risk"] = df.apply(calculate_risk, axis=1)


# ==========================================
# 6. SELECT TIMESTAMP
# ==========================================

# Change this value to test:
#
# 300 -> Normal
# 750 -> Early degradation
# 810 -> Developing degradation
# 950 -> Severe failure

target_timestamp = 810

target_index = df.index[
    df["timestamp"] == target_timestamp
][0]


# ==========================================
# 7. GET CURRENT DATA
# ==========================================

latest = df.loc[target_index]

temperature = latest["temperature"]
current = latest["current"]
vibration = latest["vibration"]
pressure = latest["pressure"]

risk = latest["risk"]

anomaly = latest["anomaly"]


# ==========================================
# 8. TREND ANALYSIS
# ==========================================

# Compare the previous 50 readings with
# the current 50 readings.

current_start = max(0, target_index - 49)

previous_start = max(0, target_index - 99)
previous_end = max(0, target_index - 49)

current_window = df.loc[
    current_start:target_index,
    "risk"
]

previous_window = df.loc[
    previous_start:previous_end,
    "risk"
]

current_average = current_window.mean()

previous_average = previous_window.mean()

risk_change = current_average - previous_average


# ==========================================
# 9. DETERMINE TREND
# ==========================================

if risk_change > 20:

    trend = "STRONGLY INCREASING"

elif risk_change > 5:

    trend = "INCREASING"

elif risk_change < -5:

    trend = "DECREASING"

else:

    trend = "STABLE"


# ==========================================
# 10. FINAL STATUS
# ==========================================

if risk >= 70:

    status = "HIGH_FAILURE_RISK"

elif risk >= 40:

    status = "EARLY_DEGRADATION"

elif trend in ["INCREASING", "STRONGLY INCREASING"]:

    status = "MONITOR_CLOSELY"

else:

    status = "NORMAL"


# ==========================================
# 11. DISPLAY RESULT
# ==========================================

print("\n==========================================")
print("        SILENT FAILURE DETECTOR")
print("        AI PREDICTION PIPELINE")
print("==========================================\n")

print(f"Timestamp        : {target_timestamp}")

print("\nSensor Values")
print("------------------------------------------")

print(f"Temperature      : {temperature:.2f}")
print(f"Current          : {current:.2f}")
print(f"Vibration        : {vibration:.2f}")
print(f"Pressure         : {pressure:.2f}")

print("\nAI Analysis")
print("------------------------------------------")

print(f"Anomaly          : {anomaly}")

print(f"Failure Risk     : {risk:.2f}/100")

print(f"Risk Trend       : {trend}")

print(f"Risk Change      : {risk_change:.2f}")

print(f"Final Status     : {status}")

print("\n==========================================")


# ==========================================
# 12. RECOMMENDATION
# ==========================================

if status == "HIGH_FAILURE_RISK":

    print("🚨 HIGH FAILURE RISK")

    print(
        "Multiple sensor readings indicate "
        "significant equipment degradation."
    )

    print(
        "ACTION: Inspect the equipment immediately."
    )


elif status == "EARLY_DEGRADATION":

    print("⚠️ EARLY DEGRADATION DETECTED")

    print(
        "The machine is showing abnormal behavior "
        "before severe failure."
    )

    print(
        "ACTION: Schedule inspection and continue monitoring."
    )


elif status == "MONITOR_CLOSELY":

    print("🟡 MONITOR CLOSELY")

    print(
        "Risk is increasing even though the "
        "current risk level is not yet high."
    )

    print(
        "ACTION: Continue monitoring the machine."
    )


else:

    print("🟢 SYSTEM NORMAL")

    print(
        "Machine behavior is currently "
        "within the expected range."
    )

print("\n==========================================")