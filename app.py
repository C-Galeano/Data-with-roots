from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

df = pd.read_csv("data/coffee_shop_revenue.csv")

x = df[["Marketing_Spend_Per_Day"]]
y = df[["Number_of_Customers_Per_Day"]]

total_records = len(df)

model = LinearRegression()
model.fit(x, y)


# --- Supervised: Logistic Regression (data setup) -----------------------
# Topic: Seed Germination Prediction
# Independent variable: Soil_Moisture_Percent
# Target variable (binary): Germinated (0 = No, 1 = Yes)

logreg_df = pd.read_csv("data/seed_germination.csv")

logreg_total_records = len(logreg_df)

logreg_x = logreg_df[["Soil_Moisture_Percent"]]
logreg_y = logreg_df["Germinated"]

logreg_x_train, logreg_x_test, logreg_y_train, logreg_y_test = train_test_split(
    logreg_x, logreg_y, test_size=0.2, random_state=42
)

logreg_model = LogisticRegression()
logreg_model.fit(logreg_x_train, logreg_y_train)


def classifyGermination(soil_moisture):
    input_df = pd.DataFrame({"Soil_Moisture_Percent": [soil_moisture]})
    predicted_class = int(logreg_model.predict(input_df)[0])
    probability = logreg_model.predict_proba(input_df)[0][1]
    return predicted_class, round(probability * 100, 1)


def create_logreg_plot(soil_moisture=None, predicted_class=None):
    plt.figure(figsize=(8, 5))

    germinated = logreg_df[logreg_df["Germinated"] == 1]
    not_germinated = logreg_df[logreg_df["Germinated"] == 0]

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
    probability_curve = logreg_model.predict_proba(moisture_range)[:, 1]

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

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()

    return plot_url


def calculateCustomers(marketing_spend):
    result = model.predict([[marketing_spend]])[0][0]
    return round(result)

def create_plot(marketing_spend=None, predicted_customers=None):
    plt.figure(figsize=(8, 5))

    # Mostrar 500 datos reales del dataset
    x_plot = x.head(500)
    y_plot = y.head(500)

    # Datos reales (con transparencia para que no se amontonen)
    plt.scatter(
        x_plot["Marketing_Spend_Per_Day"],
        y_plot["Number_of_Customers_Per_Day"],
        alpha=0.4,
        color="steelblue",
        label="Datos reales"
    )

    # Rango completo de X del dataset (ordenado para trazar bien la recta)
    x_line = pd.DataFrame({
        "Marketing_Spend_Per_Day": [
            x["Marketing_Spend_Per_Day"].min(),
            x["Marketing_Spend_Per_Day"].max()
        ]
    })

    # Valores Y calculados por el modelo de regresión lineal
    y_line = model.predict(x_line)

    # Recta de regresión (más gruesa y en color contrastante)
    plt.plot(
        x_line["Marketing_Spend_Per_Day"],
        y_line,
        color="red",
        linewidth=2.5,
        label="Recta de regresión"
    )

    # Predicción realizada por el usuario (marcador distinto y visible)
    if marketing_spend is not None and predicted_customers is not None:
        plt.scatter(
            marketing_spend,
            predicted_customers,
            s=180,
            color="black",
            marker="X",
            zorder=5,
            label="Tu predicción"
        )

    plt.xlabel("Marketing Spend Per Day")
    plt.ylabel("Number of Customers Per Day")
    plt.title("Linear Regression: Marketing Spend vs Number of Customers")
    plt.legend()
    plt.grid(alpha=0.3)

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()

    return plot_url


# --- Home -------------------------------------------------------------

@app.route("/")
def home():
    return render_template("home.html")


# --- Machine Learning ---------------------------------------------------

@app.route("/ml/concepts")
def ml_concepts():
    return render_template("ml/concepts.html")


@app.route("/ml/types")
def ml_types():
    return render_template("ml/types.html")


# --- Use Cases ------------------------------------------------------------

@app.route("/use-cases/1")
def use_case_1():
    return render_template("use_cases/use_case_1.html")


@app.route("/use-cases/2")
def use_case_2():
    return render_template("use_cases/use_case_2.html")


@app.route("/use-cases/3")
def use_case_3():
    return render_template("use_cases/use_case_3.html")


@app.route("/use-cases/4")
def use_case_4():
    return render_template("use_cases/use_case_4.html")


# --- Supervised: Linear Regression --------------------------------------

@app.route("/regression/concepts")
def regression_concepts():
    return render_template("regression/concepts.html")


@app.route("/regression/application", methods=["GET", "POST"])
def regression_application():
    calculateResult = None
    plot_url = None

    if request.method == "POST":
        marketing_spend = request.form.get("marketing_spend")

        if marketing_spend:
            marketing_spend = float(marketing_spend)
            calculateResult = calculateCustomers(marketing_spend)

            plot_url = create_plot(
                marketing_spend,
                calculateResult
            )

    return render_template(
        "regression/application.html",
        result=calculateResult,
        plot_url=plot_url,
        total_records=total_records
    )


# --- Supervised: Logistic Regression ------------------------------------

@app.route("/logistic-regression/concepts")
def logistic_regression_concepts():
    return render_template("logistic_regression/concepts.html")


@app.route("/logistic-regression/application", methods=["GET", "POST"])
def logistic_regression_application():
    predicted_class = None
    predicted_label = None
    probability = None
    plot_url = None
    error = None

    if request.method == "POST":
        soil_moisture = request.form.get("soil_moisture")

        if soil_moisture:
            soil_moisture = float(soil_moisture)

            if 0 <= soil_moisture <= 100:
                predicted_class, probability = classifyGermination(soil_moisture)
                predicted_label = "Germinated" if predicted_class == 1 else "Not Germinated"

                plot_url = create_logreg_plot(soil_moisture, predicted_class)
            else:
                error = "Soil moisture must be a value between 0 and 100."

    return render_template(
        "logistic_regression/application.html",
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        probability=probability,
        plot_url=plot_url,
        error=error,
        total_records=logreg_total_records
    )

# --- Unsupervised Machine Learning ---

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Part 1: results of the manual K-Means simulation (kmeans_manual/manual_kmeans.py)
with open(os.path.join(BASE_DIR, "kmeans_manual", "results", "results.json")) as f:
    manual_kmeans = json.load(f)

MANUAL_CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # same colors as the plots


def build_manual_kmeans_context():
    """Prepare the manual K-Means results so the template only has to display them."""
    iterations = []
    for it in manual_kmeans["iterations"]:
        rows = []
        for row in it["rows"]:
            distances = row["distances"]
            rows.append({
                "tray_id": row["tray_id"],
                "x": row["x"],
                "y": row["y"],
                "distances": distances,
                "nearest": distances.index(min(distances)),
                "cluster": row["cluster"],
            })

        clusters = []
        for k, c in enumerate(it["clusters"]):
            clusters.append({
                "name": f"C{k + 1}",
                "color": MANUAL_CLUSTER_COLORS[k],
                "records": c["records"],
                "sse": c["sse"],
                "variance": c["variance"],
                "before": it["centroids_before"][k],
                "after": it["centroids_after"][k],
            })

        iterations.append({
            "number": it["iteration"],
            "rows": rows,
            "clusters": clusters,
            "wcss": it["wcss"],
            "reassigned": it["reassigned"],
        })

    # Final cluster profiles (ranges of the trays assigned in the last iteration)
    last = iterations[-1]
    final_clusters = []
    for k, c in enumerate(last["clusters"]):
        members = [r for r in last["rows"] if r["cluster"] == k + 1]
        final_clusters.append({
            **c,
            "x_min": min(r["x"] for r in members),
            "x_max": max(r["x"] for r in members),
            "y_min": min(r["y"] for r in members),
            "y_max": max(r["y"] for r in members),
        })

    xs = [r["x"] for r in last["rows"]]
    ys = [r["y"] for r in last["rows"]]
    dataset_stats = {
        "records": len(xs),
        "x_min": min(xs), "x_max": max(xs), "x_mean": sum(xs) / len(xs),
        "y_min": min(ys), "y_max": max(ys), "y_mean": sum(ys) / len(ys),
    }

    wcss_first = iterations[0]["wcss"]
    wcss_last = iterations[-1]["wcss"]

    return {
        "initial_centroids": manual_kmeans["initial_centroids"],
        "colors": MANUAL_CLUSTER_COLORS,
        "iterations": iterations,
        "final_clusters": final_clusters,
        "dataset_stats": dataset_stats,
        "wcss_reduction": (wcss_first - wcss_last) / wcss_first * 100,
    }


@app.route("/unsupervised/concepts")
def unsupervised_concepts():
    return render_template("unsupervised/concepts.html")


@app.route("/unsupervised/manual-exercise")
def unsupervised_manual():
    return render_template("unsupervised/manual.html", **build_manual_kmeans_context())


@app.route("/unsupervised/clustering")
def unsupervised_clustering():
    # Clustering Application (Scikit-learn K-Means) - in progress by team member 3
    return render_template("unsupervised/clustering.html")

if __name__ == "__main__":
    app.run(debug=True)
