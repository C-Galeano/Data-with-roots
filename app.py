from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

df = pd.read_csv("data/coffee_shop_revenue.csv")

x = df[["Marketing_Spend_Per_Day"]]
y = df[["Number_of_Customers_Per_Day"]]

total_records = len(df)

model = LinearRegression()
model.fit(x, y)


# --- Supervised: Logistic Regression (data setup) ---

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


# --- Supervised: Gradient Boosting Classifier ----
GBC_FEATURES = [
    "Temperature_C",
    "Humidity_Percent",
    "Rainfall_mm",
    "Soil_Nitrogen_Level",
]

gbc_df = pd.read_csv("data/crop_disease_risk.csv")

gbc_total_records = len(gbc_df)

gbc_x = gbc_df[GBC_FEATURES]
gbc_y = gbc_df["Disease_Risk"]

gbc_x_train, gbc_x_test, gbc_y_train, gbc_y_test = train_test_split(
    gbc_x, gbc_y, test_size=0.2, random_state=42
)

gbc_model = GradientBoostingClassifier(random_state=42)
gbc_model.fit(gbc_x_train, gbc_y_train)


def classifyDiseaseRisk(temperature, humidity, rainfall, nitrogen):
    input_df = pd.DataFrame({
        "Temperature_C": [temperature],
        "Humidity_Percent": [humidity],
        "Rainfall_mm": [rainfall],
        "Soil_Nitrogen_Level": [nitrogen],
    })
    predicted_class = int(gbc_model.predict(input_df)[0])
    probability = gbc_model.predict_proba(input_df)[0][1]
    return predicted_class, round(probability * 100, 1)


def create_gbc_plot(temperature=None, humidity=None, predicted_class=None):
    plt.figure(figsize=(8, 5))

    low_risk = gbc_df[gbc_df["Disease_Risk"] == 0]
    high_risk = gbc_df[gbc_df["Disease_Risk"] == 1]

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

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()

    return plot_url


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

    x_plot = x.head(500)
    y_plot = y.head(500)

    plt.scatter(
        x_plot["Marketing_Spend_Per_Day"],
        y_plot["Number_of_Customers_Per_Day"],
        alpha=0.4,
        color="steelblue",
        label="Datos reales"
    )

    x_line = pd.DataFrame({
        "Marketing_Spend_Per_Day": [
            x["Marketing_Spend_Per_Day"].min(),
            x["Marketing_Spend_Per_Day"].max()
        ]
    })

    y_line = model.predict(x_line)

    plt.plot(
        x_line["Marketing_Spend_Per_Day"],
        y_line,
        color="red",
        linewidth=2.5,
        label="Recta de regresión"
    )

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


# --- Home -------

@app.route("/")
def home():
    return render_template("home.html")


# --- Machine Learning ------

@app.route("/ml/concepts")
def ml_concepts():
    return render_template("ml/concepts.html")


@app.route("/ml/types")
def ml_types():
    return render_template("ml/types.html")


# --- Use Cases ------

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


# --- Supervised: Linear Regression --

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


# --- Supervised: Logistic Regression --

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


# --- Supervised: Gradient Boosting Classifier ----

@app.route("/gradient-boosting/concepts")
def gradient_boosting_concepts():
    return render_template("gradient_boosting/concepts.html")


@app.route("/gradient-boosting/application", methods=["GET", "POST"])
def gradient_boosting_application():
    predicted_class = None
    predicted_label = None
    probability = None
    plot_url = None
    error = None
    temperature = None
    humidity = None

    if request.method == "POST":
        temperature = request.form.get("temperature")
        humidity = request.form.get("humidity")
        rainfall = request.form.get("rainfall")
        nitrogen = request.form.get("nitrogen")

        if temperature and humidity and rainfall and nitrogen:
            temperature = float(temperature)
            humidity = float(humidity)
            rainfall = float(rainfall)
            nitrogen = float(nitrogen)

            if 0 <= humidity <= 100 and 0 <= nitrogen <= 100 and rainfall >= 0:
                predicted_class, probability = classifyDiseaseRisk(
                    temperature, humidity, rainfall, nitrogen
                )
                predicted_label = "High Risk" if predicted_class == 1 else "Low Risk"

                plot_url = create_gbc_plot(temperature, humidity, predicted_class)
            else:
                error = "Humidity and Nitrogen must be between 0 and 100, and Rainfall must be 0 or greater."

    return render_template(
        "gradient_boosting/application.html",
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        probability=probability,
        plot_url=plot_url,
        error=error,
        total_records=gbc_total_records
    )


if __name__ == "__main__":
    app.run(debug=True)