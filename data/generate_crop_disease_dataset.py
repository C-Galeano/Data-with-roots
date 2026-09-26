import csv
import math
import random

random.seed(42)

RECORDS = 700

rows = []
for _ in range(RECORDS):
    temperature = round(random.uniform(10, 40), 1)
    humidity = round(random.uniform(20, 100), 1)
    rainfall = round(random.uniform(0, 200), 1)
    nitrogen = round(random.uniform(0, 100), 1)

    z = (
        0.045 * (temperature - 25)
        + 0.05 * (humidity - 55)
        + 0.02 * (rainfall - 60)
        - 0.045 * (nitrogen - 50)
    )
    probability = 1 / (1 + math.exp(-z))
    disease_risk = 1 if random.random() < probability else 0

    rows.append((temperature, humidity, rainfall, nitrogen, disease_risk))

with open("data/crop_disease_risk.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Temperature_C",
        "Humidity_Percent",
        "Rainfall_mm",
        "Soil_Nitrogen_Level",
        "Disease_Risk"
    ])
    writer.writerows(rows)

print(f"Wrote {len(rows)} records to data/crop_disease_risk.csv")