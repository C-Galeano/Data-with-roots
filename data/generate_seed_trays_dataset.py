"""
Generates data/seed_trays.csv (manual K-Means exercise, Part 1).

Context: 100 seed trays in a nursery, measured 14 days after sowing.
Variable 1 : Soil_Moisture_Percent    (average soil moisture of the tray, %).
Variable 2 : Germination_Rate_Percent (share of seeds that germinated, %).

Three growing conditions are simulated so natural groups exist in the data:
  - under-watered trays : dry substrate, few seeds germinate.
  - well-watered trays  : moisture in the optimal range, high germination.
  - over-watered trays  : waterlogged substrate, seeds rot, germination drops.
Gaussian noise makes the groups overlap slightly, as in a real nursery.

Both variables are percentages (0-100), so they share the same scale and the
Euclidean distance can be computed on the raw values without normalization.

Run once with: python data/generate_seed_trays_dataset.py
"""

import csv
import random

random.seed(7)

# (records, mean moisture, sd moisture, mean germination, sd germination)
GROUPS = [
    (34, 22.0, 5.0, 28.0, 7.0),   # under-watered
    (36, 47.0, 5.5, 84.0, 6.0),   # well-watered
    (30, 74.0, 5.0, 52.0, 7.5),   # over-watered
]


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


points = []
for records, mx, sx, my, sy in GROUPS:
    for _ in range(records):
        moisture = round(clamp(random.gauss(mx, sx)), 1)
        germination = round(clamp(random.gauss(my, sy)), 1)
        points.append((moisture, germination))

random.shuffle(points)

with open("data/seed_trays.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Tray_ID", "Soil_Moisture_Percent", "Germination_Rate_Percent"])
    for i, (moisture, germination) in enumerate(points, start=1):
        writer.writerow([f"T{i:03d}", moisture, germination])

print(f"Wrote {len(points)} records to data/seed_trays.csv")
