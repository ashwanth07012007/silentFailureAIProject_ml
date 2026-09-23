import pandas as pd
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv("data/sensor_data.csv")

# Sensors used by the AI
features = [
    "temperature",
    "current",
    "vibration",
    "pressure"
]

X = df[features]

# Create the AI model
model = IsolationForest(
    contamination=0.1,
    random_state=42
)

# Train the model
model.fit(X)

# Predict anomalies
df["prediction"] = model.predict(X)

# Convert result into readable form
df["status"] = df["prediction"].map({
    1: "NORMAL",
    -1: "ANOMALY"
})

print(df[[
    "timestamp",
    "temperature",
    "current",
    "vibration",
    "pressure",
    "status"
]].tail(100))

# Count results
print("\nDetection summary:")
print(df["status"].value_counts())