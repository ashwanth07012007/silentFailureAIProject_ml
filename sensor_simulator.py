import requests
import time
import random


# ==========================================
# SPRING BOOT API
# ==========================================

API_URL = "https://silentfailureaiproject-backend.onrender.com/api/sensors"


# ==========================================
# STARTING SENSOR VALUES
# ==========================================

temperature = 30.0
current = 5.0
vibration = 2.0
pressure = 5.0


# ==========================================
# SIMULATION SETTINGS
# ==========================================

reading_number = 0


print("\n==========================================")
print("      SILENT FAILURE SENSOR SIMULATOR")
print("==========================================")
print("Sending data every 5 seconds...")
print("Press CTRL+C to stop.\n")


while True:

    reading_number += 1


    # ======================================
    # NORMAL OPERATION
    # ======================================

    if reading_number <= 20:

        temperature += random.uniform(-0.2, 0.2)

        current += random.uniform(-0.03, 0.03)

        vibration += random.uniform(-0.05, 0.05)

        pressure += random.uniform(-0.03, 0.03)


        stage = "NORMAL"


    # ======================================
    # GRADUAL DEGRADATION
    # ======================================

    elif reading_number <= 50:

        temperature += random.uniform(0.15, 0.35)

        current += random.uniform(0.01, 0.04)

        vibration += random.uniform(0.03, 0.08)

        pressure -= random.uniform(0.02, 0.05)


        stage = "DEGRADATION"


    # ======================================
    # SEVERE FAILURE
    # ======================================

    else:

        temperature += random.uniform(0.3, 0.6)

        current += random.uniform(0.03, 0.08)

        vibration += random.uniform(0.08, 0.15)

        pressure -= random.uniform(0.04, 0.08)


        stage = "SEVERE FAILURE"


    # ======================================
    # PREVENT UNREALISTIC VALUES
    # ======================================

    temperature = max(20, temperature)

    current = max(3, current)

    vibration = max(0.5, vibration)

    pressure = max(1, pressure)


    # ======================================
    # CREATE SENSOR DATA
    # ======================================

    sensor_data = {

        "temperature": round(temperature, 2),

        "current": round(current, 2),

        "vibration": round(vibration, 2),

        "pressure": round(pressure, 2)

    }


    # ======================================
    # SEND TO SPRING BOOT
    # ======================================

    try:

        response = requests.post(
            API_URL,
            json=sensor_data,
            timeout=5
        )


        if response.status_code == 200:

            result = response.json()


            print(
                f"Reading {reading_number:03d} | "
                f"Stage: {stage}"
            )

            print(
                f"Temp: {sensor_data['temperature']:6.2f} | "
                f"Current: {sensor_data['current']:5.2f} | "
                f"Vibration: {sensor_data['vibration']:5.2f} | "
                f"Pressure: {sensor_data['pressure']:5.2f}"
            )

            print(
                f"Risk: {result['risk']:6.2f} | "
                f"Anomaly: {result['anomaly']} | "
                f"Status: {result['status']}"
            )

            print(
                f"Trend: {result['riskTrend']}"
            )

            print(
                f"Recommendation: "
                f"{result['recommendation']}"
            )

            print("-" * 70)


        else:

            print(
                "Spring Boot returned:",
                response.status_code
            )

            print(response.text)


    except requests.exceptions.RequestException as error:

        print("\n❌ Could not connect to Spring Boot")

        print(error)


    # ======================================
    # WAIT 5 SECONDS
    # ======================================

    time.sleep(5)