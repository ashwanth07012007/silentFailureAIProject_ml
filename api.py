from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Silent Failure AI API is running!"


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/sensor_data.csv")

features = [
    "temperature",
    "current",
    "vibration",
    "pressure"
]


# ==========================================
# HEALTHY BASELINE
# ==========================================

normal = df[df["timestamp"] < 700]

normal_values = {
    "temperature": normal["temperature"].mean(),
    "current": normal["current"].mean(),
    "vibration": normal["vibration"].mean(),
    "pressure": normal["pressure"].mean()
}


# ==========================================
# TRAIN ISOLATION FOREST
# ONLY USING HEALTHY DATA
# ==========================================

X_normal = normal[features]

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X_normal)


# ==========================================
# LIVE RISK HISTORY
# ==========================================

risk_history = []

MAX_HISTORY = 10


# ==========================================
# SENSOR SCORE
# ==========================================

def sensor_score(value, normal_value, limit):

    deviation = abs(value - normal_value) / normal_value

    score = deviation / limit

    return min(score, 1)


# ==========================================
# CALCULATE RISK
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
        temperature_score * 25
        + current_score * 25
        + vibration_score * 25
        + pressure_score * 25
    )

    return min(risk, 100)


# ==========================================
# TREND DETECTION
# ==========================================

def calculate_trend():

    # Not enough readings yet
    if len(risk_history) < 4:
        return "INSUFFICIENT_DATA", 0, 0, 0

    midpoint = len(risk_history) // 2

    earlier = risk_history[:midpoint]

    recent = risk_history[midpoint:]

    earlier_average = sum(earlier) / len(earlier)

    recent_average = sum(recent) / len(recent)

    risk_change = recent_average - earlier_average


    if risk_change > 20:

        trend = "STRONGLY_INCREASING"

    elif risk_change > 5:

        trend = "INCREASING"

    elif risk_change < -5:

        trend = "DECREASING"

    else:

        trend = "STABLE"


    return (
        trend,
        earlier_average,
        recent_average,
        risk_change
    )


# ==========================================
# PREDICT API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()


    # --------------------------------------
    # GET SENSOR VALUES
    # --------------------------------------

    temperature = float(data["temperature"])

    current = float(data["current"])

    vibration = float(data["vibration"])

    pressure = float(data["pressure"])


    # --------------------------------------
    # ISOLATION FOREST
    # --------------------------------------

    sensor_data = pd.DataFrame(
        [[
            temperature,
            current,
            vibration,
            pressure
        ]],
        columns=features
    )

    prediction = model.predict(sensor_data)[0]

    anomaly = prediction == -1


    # --------------------------------------
    # RISK
    # --------------------------------------

    risk = calculate_risk(
        temperature,
        current,
        vibration,
        pressure
    )


    # --------------------------------------
    # STORE RISK
    # --------------------------------------

    risk_history.append(risk)


    # Keep only latest 10 readings

    if len(risk_history) > MAX_HISTORY:

        risk_history.pop(0)


    # --------------------------------------
    # TREND
    # --------------------------------------

    (
        trend,
        earlier_average,
        recent_average,
        risk_change
    ) = calculate_trend()


    # --------------------------------------
    # STATUS
    # --------------------------------------

    if risk >= 70:

        status = "HIGH_FAILURE_RISK"

    elif risk >= 40:

        status = "EARLY_DEGRADATION"

    elif anomaly:

        status = "EARLY_DEGRADATION"

    elif trend in [
        "INCREASING",
        "STRONGLY_INCREASING"
    ]:

        status = "MONITOR_CLOSELY"

    else:

        status = "NORMAL"


    # --------------------------------------
    # RECOMMENDATION
    # --------------------------------------

    if status == "HIGH_FAILURE_RISK":

        recommendation = (
            "Immediate inspection recommended."
        )

    elif status == "EARLY_DEGRADATION":

        recommendation = (
            "Schedule inspection and continue monitoring."
        )

    elif status == "MONITOR_CLOSELY":

        recommendation = (
            "Risk is increasing. Monitor equipment closely."
        )

    else:

        recommendation = (
            "System operating within normal behavior."
        )


    # ======================================
    # RESPONSE
    # ======================================

    result = {

        "temperature": round(temperature, 2),

        "current": round(current, 2),

        "vibration": round(vibration, 2),

        "pressure": round(pressure, 2),

        "anomaly": bool(anomaly),

        "risk": round(risk, 2),

        "riskHistory": [
            round(value, 2)
            for value in risk_history
        ],

        "earlyAvgRisk": round(
            earlier_average, 2
        ),

        "recentAvgRisk": round(
            recent_average, 2
        ),

        "riskChange": round(
            risk_change, 2
        ),

        "riskTrend": trend,

        "status": status,

        "recommendation": recommendation
    }


    return jsonify(result)


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )