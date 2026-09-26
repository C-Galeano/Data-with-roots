"""
Supervised: Logistic Regression.

Topic: Seed Germination Prediction.
Independent variable      : Soil_Moisture_Percent
Target variable (binary)  : Germinated (0 = No, 1 = Yes)
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.metrics import classification_metrics
from utils.paths import DATA_DIR
from utils.plots import create_confusion_matrix_plot, fig_to_base64

df = pd.read_csv(os.path.join(DATA_DIR, "seed_germination.csv"))

total_records = len(df)

x = df[["Soil_Moisture_Percent"]]
y = df["Germinated"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(x_train, y_train)

metrics = classification_metrics(y_test, model.predict(x_test))


def classify_germination(soil_moisture):
    input_df = pd.DataFrame({"Soil_Moisture_Percent": [soil_moisture]})
    predicted_class = int(model.predict(input_df)[0])
    probability = model.predict_proba(input_df)[0][1]
    return predicted_class, round(probability * 100, 1)


def create_plot(soil_moisture=None, predicted_class=None):
    plt.figure(figsize=(8, 5))

    germinated = df[df["Germinated"] == 1]
    not_germinated = df[df["Germinated"] == 0]

    plt.scatter(
        not_germinated["Soil_Moisture_Percent"],
        not_germinated["Germinated"],
        alpha=0.5,
        color="firebrick",
        label="Not Germinated (0)"
    )
    plt.scatter(
        germinated["Soil_Moisture_Percent"],
        germinated["Germinated"],
        alpha=0.5,
        color="seagreen",
        label="Germinated (1)"
    )

    moisture_range = pd.DataFrame({
        "Soil_Moisture_Percent": [i / 2 for i in range(10, 191)]
    })
    probability_curve = model.predict_proba(moisture_range)[:, 1]

    plt.plot(
        moisture_range["Soil_Moisture_Percent"],
        probability_curve,
        color="steelblue",
        linewidth=2.5,
        label="Predicted probability of germination"
    )

    if soil_moisture is not None and predicted_class is not None:
        plt.scatter(
            soil_moisture,
            predicted_class,
            s=180,
            color="black",
            marker="X",
            zorder=5,
            label="Your prediction"
        )

    plt.xlabel("Soil Moisture (%)")
    plt.ylabel("Germinated (0 = No, 1 = Yes)")
    plt.title("Logistic Regression: Soil Moisture vs Seed Germination")
    plt.legend(loc="center right")
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
            ["Not Germinated (0)", "Germinated (1)"],
            "Logistic Regression - Confusion Matrix"
        ),
    }
