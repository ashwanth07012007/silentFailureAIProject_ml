import pandas as pd

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
# 3. SENSOR RISK FUNCTION
# ==========================================

def sensor_score(value, normal_value, limit):

    deviation = abs(value - normal_value) / normal_value

    score = deviation / limit

    return min(score, 1)


# ==========================================
# 4. CALCULATE RISK
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


# Calculate risk for every row

df["risk"] = df.apply(calculate_risk, axis=1)


# ==========================================
# 5. SMOOTH THE RISK
# ==========================================

window = 30

df["rolling_risk"] = (
    df["risk"]
    .rolling(window)
    .mean()
)


# ==========================================
# 6. COMPARE EARLIER AND RECENT PERIODS
# ==========================================

# Earlier degradation period
earlier = df[
    (df["timestamp"] >= 700) &
    (df["timestamp"] < 750)
]

# More recent degradation period
recent = df[
    (df["timestamp"] >= 850) &
    (df["timestamp"] < 900)
]


earlier_risk = earlier["rolling_risk"].mean()

recent_risk = recent["rolling_risk"].mean()

risk_change = recent_risk - earlier_risk


# ==========================================
# 7. DETERMINE TREND
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
# 8. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("       SILENT FAILURE DETECTOR")
print("       TREND ANALYSIS")
print("==========================================\n")

print(f"Earlier Risk : {earlier_risk:.2f}/100")

print(f"Recent Risk  : {recent_risk:.2f}/100")

print(f"Risk Change  : {risk_change:.2f}")

print(f"Risk Trend   : {trend}")


# ==========================================
# 9. EARLY WARNING
# ==========================================

print("\n------------------------------------------")


if trend == "STRONGLY INCREASING":

    print("🚨 EARLY FAILURE WARNING")

    print(
        "System risk has increased significantly "
        "over time."
    )

    print(
        "The machine may be experiencing "
        "gradual degradation."
    )

    print(
        "Recommended action: Inspect the equipment."
    )


elif trend == "INCREASING":

    print("⚠️ WARNING")

    print(
        "System risk is gradually increasing."
    )

    print(
        "Continue monitoring the equipment."
    )


else:

    print("🟢 NO EARLY WARNING")

    print(
        "No significant risk increase detected "
        "between the analyzed periods."
    )