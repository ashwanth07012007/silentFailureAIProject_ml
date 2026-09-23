import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1000

data = []

for i in range(rows):

    # Normal sensor behavior
    temperature = np.random.normal(30, 1)
    current = np.random.normal(5, 0.2)
    vibration = np.random.normal(2, 0.2)
    pressure = np.random.normal(5, 0.2)

    # Gradual degradation
    if i >= 700:
        temperature += (i - 700) * 0.03
        current += (i - 700) * 0.005
        vibration += (i - 700) * 0.01
        pressure -= (i - 700) * 0.005

    # Severe failure behavior
    if i >= 900:
        temperature += 15
        current += 2
        vibration += 3
        pressure -= 1

    data.append([
        i,
        temperature,
        current,
        vibration,
        pressure
    ])


df = pd.DataFrame(
    data,
    columns=[
        "timestamp",
        "temperature",
        "current",
        "vibration",
        "pressure"
    ]
)

df.to_csv("data/sensor_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
print("\nDataset shape:", df.shape)