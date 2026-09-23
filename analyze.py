import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv("data/sensor_data.csv")

features = [
    "temperature",
    "current",
    "vibration",
    "pressure"
]

X = df[features]

# Train model
model = IsolationForest(
    contamination=0.1,
    random_state=42
)

model.fit(X)

# Predict
df["prediction"] = model.predict(X)

df["status"] = df["prediction"].map({
    1: "NORMAL",
    -1: "ANOMALY"
})

# Look at the latest sensor reading
latest = df.iloc[-1]

print("\n========== SILENT FAILURE DETECTOR ==========\n")

print(f"Timestamp    : {latest['timestamp']}")
print(f"Temperature  : {latest['temperature']:.2f}")
print(f"Current      : {latest['current']:.2f}")
print(f"Vibration    : {latest['vibration']:.2f}")
print(f"Pressure     : {latest['pressure']:.2f}")

print("\nSystem status:", latest["status"])

if latest["status"] == "ANOMALY":

    print("\n⚠️ ANOMALY DETECTED")

    # Compare with normal operating region
    normal = df[df["status"] == "NORMAL"]

    if latest["temperature"] > normal["temperature"].mean():
        print("🌡️ Temperature is unusually HIGH")

    if latest["current"] > normal["current"].mean():
        print("⚡ Current is unusually HIGH")

    if latest["vibration"] > normal["vibration"].mean():
        print("📳 Vibration is unusually HIGH")

    if latest["pressure"] < normal["pressure"].mean():
        print("💧 Pressure is unusually LOW")

    print("\nPossible issue:")
    print("Machine behavior indicates potential degradation.")

    print("\nRecommended action:")
    print("Inspect the equipment before complete failure.")

else:

    print("\n🟢 System operating normally")