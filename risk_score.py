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

X = df[features]


# ==========================================
# 2. ANOMALY DETECTION
# ==========================================

model = IsolationForest(
    contamination=0.1,
    random_state=42
)

model.fit(X)

df["prediction"] = model.predict(X)


# ==========================================
# 3. NORMAL BASELINE
# ==========================================

# First 700 records represent normal behavior

normal = df[df["timestamp"] < 700]

normal_temperature = normal["temperature"].mean()
normal_current = normal["current"].mean()
normal_vibration = normal["vibration"].mean()
normal_pressure = normal["pressure"].mean()


# ==========================================
# 4. SELECT TIMESTAMP
# ==========================================

# Test:
# 300 = Normal
# 750 = Early degradation
# 810 = Developing degradation
# 950 = Severe failure

latest = df.iloc[850]


temperature = latest["temperature"]
current = latest["current"]
vibration = latest["vibration"]
pressure = latest["pressure"]


# ==========================================
# 5. CALCULATE DEVIATION
# ==========================================

temp_deviation = (
    abs(temperature - normal_temperature)
    / normal_temperature
)

current_deviation = (
    abs(current - normal_current)
    / normal_current
)

vibration_deviation = (
    abs(vibration - normal_vibration)
    / normal_vibration
)

pressure_deviation = (
    abs(pressure - normal_pressure)
    / normal_pressure
)


# ==========================================
# 6. NORMALIZE DEVIATIONS
# ==========================================
# These values represent approximately how much
# deviation should contribute to high risk.

temp_score = min(temp_deviation / 0.40, 1)
current_score = min(current_deviation / 0.40, 1)
vibration_score = min(vibration_deviation / 1.00, 1)
pressure_score = min(pressure_deviation / 0.50, 1)


# ==========================================
# 7. COMBINE SENSOR SCORES
# ==========================================

risk = (
    temp_score * 25 +
    current_score * 25 +
    vibration_score * 25 +
    pressure_score * 25
)

risk = min(risk, 100)


# ==========================================
# 8. RISK LEVEL
# ==========================================

if risk >= 70:

    level = "HIGH"

elif risk >= 40:

    level = "MEDIUM"

else:

    level = "LOW"


# ==========================================
# 9. DISPLAY RESULT
# ==========================================

print("\n===================================")
print("      SILENT FAILURE DETECTOR")
print("===================================\n")

print(f"Timestamp   : {latest['timestamp']}")

print(f"Temperature : {temperature:.2f}")
print(f"Current     : {current:.2f}")
print(f"Vibration   : {vibration:.2f}")
print(f"Pressure    : {pressure:.2f}")

print("\n-----------------------------------")

print(f"Failure Risk : {risk:.2f}/100")
print(f"Risk Level   : {level}")

print("-----------------------------------")


# ==========================================
# 10. EXPLANATION
# ==========================================

if level == "HIGH":

    print("\n🚨 HIGH FAILURE RISK")

    print(
        "Multiple sensor values show significant "
        "deviation from normal machine behavior."
    )

    print(
        "Recommended action: Inspect the equipment "
        "before complete failure."
    )


elif level == "MEDIUM":

    print("\n⚠️ MEDIUM FAILURE RISK")

    print(
        "Sensor behavior is moving away from the "
        "normal operating condition."
    )

    print(
        "Recommended action: Continue monitoring "
        "the equipment."
    )


else:

    print("\n🟢 LOW FAILURE RISK")

    print(
        "Sensor behavior is currently close to "
        "the normal operating condition."
    )