import pandas as pd
from sklearn.ensemble import IsolationForest


# ==========================================
# 1. LOAD HISTORICAL DATA
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

# First 700 readings represent healthy behavior

normal = df[df["timestamp"] < 700]

normal_values = {
    "temperature": normal["temperature"].mean(),
    "current": normal["current"].mean(),
    "vibration": normal["vibration"].mean(),
    "pressure": normal["pressure"].mean()
}


# ==========================================
# 3. TRAIN ANOMALY DETECTOR
# ==========================================

# Train only using healthy data

X_normal = normal[features]

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X_normal)


# ==========================================
# 4. SENSOR RISK SCORE FUNCTION
# ==========================================

def sensor_score(value, normal_value, limit):

    deviation = abs(value - normal_value) / normal_value

    score = deviation / limit

    return min(score, 1)


# ==========================================
# 5. CALCULATE RISK
# ==========================================

def calculate_risk(
    temperature,
    current,
    vibration,
    pressure
):

    temperature_score = sensor_score(
        temperature,
        normal_values["temperature"],
        0.40
    )

    current_score = sensor_score(
        current,
        normal_values["current"],
        0.40
    )

    vibration_score = sensor_score(
        vibration,
        normal_values["vibration"],
        1.00
    )

    pressure_score = sensor_score(
        pressure,
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


# ==========================================
# 6. GET MULTIPLE LIVE SENSOR READINGS
# ==========================================

print("\n==========================================")
print("       SILENT FAILURE DETECTOR")
print("       LIVE SENSOR ANALYSIS")
print("==========================================")

print("\nEnter 5 consecutive sensor readings.")
print("This allows the system to detect risk trends.\n")


live_readings = []


for i in range(5):

    print("------------------------------------------")
    print(f"Reading {i + 1}")
    print("------------------------------------------")

    temperature = float(
        input("Enter temperature: ")
    )

    current = float(
        input("Enter current: ")
    )

    vibration = float(
        input("Enter vibration: ")
    )

    pressure = float(
        input("Enter pressure: ")
    )

    live_readings.append({
        "temperature": temperature,
        "current": current,
        "vibration": vibration,
        "pressure": pressure
    })


# ==========================================
# 7. ANALYZE EACH LIVE READING
# ==========================================

results = []


for reading in live_readings:

    temperature = reading["temperature"]
    current = reading["current"]
    vibration = reading["vibration"]
    pressure = reading["pressure"]


    # Create DataFrame for Isolation Forest

    current_data = pd.DataFrame(
        [[
            temperature,
            current,
            vibration,
            pressure
        ]],
        columns=features
    )


    # --------------------------------------
    # Anomaly Detection
    # --------------------------------------

    prediction = model.predict(
        current_data
    )[0]

    anomaly = prediction == -1


    # --------------------------------------
    # Risk Calculation
    # --------------------------------------

    risk = calculate_risk(
        temperature,
        current,
        vibration,
        pressure
    )


    results.append({
        "temperature": temperature,
        "current": current,
        "vibration": vibration,
        "pressure": pressure,
        "anomaly": anomaly,
        "risk": risk
    })


# ==========================================
# 8. EXTRACT RISK VALUES
# ==========================================

risk_values = [
    result["risk"]
    for result in results
]


# ==========================================
# 9. DETECT REAL LIVE TREND
# ==========================================

# Compare the first readings with
# the recent readings.

early_risk = sum(
    risk_values[:2]
) / 2

recent_risk = sum(
    risk_values[-2:]
) / 2

risk_change = recent_risk - early_risk


if risk_change > 20:

    trend = "STRONGLY INCREASING"

elif risk_change > 5:

    trend = "INCREASING"

elif risk_change < -5:

    trend = "DECREASING"

else:

    trend = "STABLE"


# ==========================================
# 10. GET LATEST READING
# ==========================================

latest = results[-1]

temperature = latest["temperature"]
current = latest["current"]
vibration = latest["vibration"]
pressure = latest["pressure"]

risk = latest["risk"]
anomaly = latest["anomaly"]


# ==========================================
# 11. FINAL STATUS
# ==========================================

if risk >= 70:

    status = "HIGH_FAILURE_RISK"

elif risk >= 40:

    status = "EARLY_DEGRADATION"

elif anomaly:

    status = "EARLY_DEGRADATION"

elif trend in [
    "INCREASING",
    "STRONGLY INCREASING"
]:

    status = "MONITOR_CLOSELY"

else:

    status = "NORMAL"


# ==========================================
# 12. RECOMMENDATION
# ==========================================

if status == "HIGH_FAILURE_RISK":

    recommendation = (
        "Inspect the equipment immediately."
    )

elif status == "EARLY_DEGRADATION":

    recommendation = (
        "Schedule inspection and continue monitoring."
    )

elif status == "MONITOR_CLOSELY":

    recommendation = (
        "Continue monitoring for increasing risk."
    )

else:

    recommendation = (
        "Equipment is operating within expected range."
    )


# ==========================================
# 14. CREATE AI RESULT
# ==========================================

ai_result = {
    "temperature": round(temperature, 2),
    "current": round(current, 2),
    "vibration": round(vibration, 2),
    "pressure": round(pressure, 2),
    "anomaly": bool(anomaly),
    "risk": round(risk, 2),
    "earlyAvgRisk": round(early_risk, 2),
    "recentAvgRisk": round(recent_risk, 2),
    "riskChange": round(risk_change, 2),
    "riskTrend": trend,
    "status": status,
    "recommendation": recommendation
}


# ==========================================
# 15. DISPLAY RESULT
# ==========================================

print("\n==========================================")
print("             AI RESULT")
print("==========================================")

print(f"Temperature       : {temperature:.2f}")
print(f"Current           : {current:.2f}")
print(f"Vibration         : {vibration:.2f}")
print(f"Pressure          : {pressure:.2f}")

print("------------------------------------------")

print(f"Anomaly           : {anomaly}")
print(f"Failure Risk      : {risk:.2f}/100")

print(f"Early Avg Risk    : {early_risk:.2f}")
print(f"Recent Avg Risk   : {recent_risk:.2f}")
print(f"Risk Change       : {risk_change:.2f}")

print(f"Risk Trend        : {trend}")
print(f"Final Status      : {status}")

print("------------------------------------------")

print(f"Recommendation    : {recommendation}")

print("==========================================\n")


# ==========================================
# 16. JSON OUTPUT
# ==========================================

import json

print("\n========== JSON OUTPUT ==========\n")

print(
    json.dumps(
        ai_result,
        indent=4
    )
)