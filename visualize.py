import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/sensor_data.csv")

sensors = [
    "temperature",
    "current",
    "vibration",
    "pressure"
]

for sensor in sensors:

    plt.figure(figsize=(12, 6))

    plt.plot(df["timestamp"], df[sensor])

    plt.axvline(
        x=700,
        linestyle="--",
        label="Degradation starts"
    )

    plt.axvline(
        x=900,
        linestyle="--",
        label="Severe failure"
    )

    plt.title(f"{sensor.capitalize()} Over Time")

    plt.xlabel("Time")

    plt.ylabel(sensor.capitalize())

    plt.legend()

    plt.grid()

    plt.show()  