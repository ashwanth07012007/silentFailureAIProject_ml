import pandas as pd

# Load dataset
df = pd.read_csv("data/sensor_data.csv")

# Number of readings used to calculate the trend
window = 50

# Calculate rolling averages
df["temperature_avg"] = df["temperature"].rolling(window).mean()
df["current_avg"] = df["current"].rolling(window).mean()
df["vibration_avg"] = df["vibration"].rolling(window).mean()
df["pressure_avg"] = df["pressure"].rolling(window).mean()


def trend(start, end):
    if end > start:
        return "INCREASING"
    elif end < start:
        return "DECREASING"
    else:
        return "STABLE"


# Look at the latest 50 readings
latest = df.iloc[-1]

previous = df.iloc[-window]


temperature_trend = trend(
    previous["temperature_avg"],
    latest["temperature_avg"]
)

current_trend = trend(
    previous["current_avg"],
    latest["current_avg"]
)

vibration_trend = trend(
    previous["vibration_avg"],
    latest["vibration_avg"]
)

pressure_trend = trend(
    previous["pressure_avg"],
    latest["pressure_avg"]
)


print("\n========== EARLY WARNING ANALYSIS ==========\n")

print("Temperature :", temperature_trend)
print("Current     :", current_trend)
print("Vibration   :", vibration_trend)
print("Pressure    :", pressure_trend)


# Count concerning trends
risk_signals = 0

if temperature_trend == "INCREASING":
    risk_signals += 1

if current_trend == "INCREASING":
    risk_signals += 1

if vibration_trend == "INCREASING":
    risk_signals += 1

if pressure_trend == "DECREASING":
    risk_signals += 1


print("\nRisk signals:", risk_signals)


if risk_signals >= 3:

    print("\n🚨 EARLY FAILURE WARNING")

    print(
        "Multiple sensor trends indicate "
        "possible equipment degradation."
    )

elif risk_signals >= 2:

    print("\n⚠️ WARNING")

    print(
        "Some sensor trends are moving "
        "away from normal behavior."
    )

else:

    print("\n🟢 NORMAL")

    print(
        "No significant degradation pattern detected."
    )