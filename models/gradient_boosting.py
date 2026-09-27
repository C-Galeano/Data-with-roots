"""
Supervised: Gradient Boosting Classifier.

Topic: Crop Disease Risk.
Independent variables     : Temperature_C, Humidity_Percent, Rainfall_mm, Soil_Nitrogen_Level
Target variable (binary)  : Disease_Risk (0 = Low, 1 = High)
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from utils.metrics import classification_metrics
from utils.paths import DATA_DIR
from utils.plots import create_confusion_matrix_plot, fig_to_base64

FEATURES = [
    "Temperature_C",
    "Humidity_Percent",
    "Rainfall_mm",
    "Soil_Nitrogen_Level",
]

df = pd.read_csv(os.path.join(DATA_DIR, "crop_disease_risk.csv"))

total_records = len(df)

x = df[FEATURES]
y = df["Disease_Risk"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = GradientBoostingClassifier(random_state=42)
model.fit(x_train, y_train)

metrics = classification_metrics(y_test, model.predict(x_test))


def classify_disease_risk(temperature, humidity, rainfall, nitrogen):
    input_df = pd.DataFrame({
        "Temperature_C": [temperature],
        "Humidity_Percent": [humidity],
        "Rainfall_mm": [rainfall],
        "Soil_Nitrogen_Level": [nitrogen],
    })
    predicted_class = int(model.predict(input_df)[0])
    probability = model.predict_proba(input_df)[0][1]
    return predicted_class, round(probability * 100, 1)


def create_plot(temperature=None, humidity=None, predicted_class=None):
    plt.figure(figsize=(8, 5))

    low_risk = df[df["Disease_Risk"] == 0]
    high_risk = df[df["Disease_Risk"] == 1]

    plt.scatter(
        low_risk["Temperature_C"],
        low_risk["Humidity_Percent"],
        alpha=0.5,
        color="seagreen",
        label="Low Risk (0)"
    )
    plt.scatter(
        high_risk["Temperature_C"],
        high_risk["Humidity_Percent"],
        alpha=0.5,
        color="firebrick",
        label="High Risk (1)"
    )

    if temperature is not None and humidity is not None:
        plt.scatter(
            temperature,
            humidity,
            s=180,
            color="black",
            marker="X",
            zorder=5,
            label="Your prediction"
        )

    plt.xlabel("Temperature (C)")
    plt.ylabel("Humidity (%)")
    plt.title("Gradient Boosting: Temperature vs Humidity by Disease Risk")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)

    return fig_to_base64()


def evaluation_context():
    """Everything the Evaluation page shows: split sizes, metrics and confusion matrix."""
    tn, fp, fn, tp = metrics["cm"].ravel()

    return {
        "total_records": total_records,
        "train_records": len(x_train),
        "test_records": len(x_test),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "plot_url": create_confusion_matrix_plot(
            metrics["cm"],
            ["Low Risk (0)", "High Risk (1)"],
            "Gradient Boosting Classifier - Confusion Matrix"
        ),
    }
