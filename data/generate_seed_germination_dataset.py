"""
Generates data/seed_germination.csv.

Topic (registered in Teams): Seed Germination Prediction.
Independent variable : Soil_Moisture_Percent (percentage of soil moisture).
Target variable       : Germinated (0 = No, 1 = Yes).

The relationship between soil moisture and the probability of germination is
modeled with a logistic (sigmoid) curve plus random noise, so the classes
overlap realistically instead of being perfectly separable.

Run once with: python data/generate_seed_germination_dataset.py
"""

import csv
import math
import random

random.seed(42)

RECORDS = 600
THRESHOLD = 42.0   # moisture (%) around which germination odds are 50/50
SLOPE = 0.18        # steepness of the relationship

rows = []
for _ in range(RECORDS):
    moisture = round(random.uniform(5, 95), 1)
    probability = 1 / (1 + math.exp(-SLOPE * (moisture - THRESHOLD)))
    germinated = 1 if random.random() < probability else 0
    rows.append((moisture, germinated))

with open("data/seed_germination.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Soil_Moisture_Percent", "Germinated"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} records to data/seed_germination.csv")
