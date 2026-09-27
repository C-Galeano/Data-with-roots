from flask import Flask, render_template, request

from models import gradient_boosting as gbc
from models import linear_regression as linreg
from models import logistic_regression as logreg
from models import unsupervised

app = Flask(__name__)


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
    result = None
    plot_url = None

    if request.method == "POST":
        marketing_spend = request.form.get("marketing_spend")

        if marketing_spend:
            marketing_spend = float(marketing_spend)
            result = linreg.calculate_customers(marketing_spend)
            plot_url = linreg.create_plot(marketing_spend, result)

    return render_template(
        "regression/application.html",
        result=result,
        plot_url=plot_url,
        total_records=linreg.total_records
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
                predicted_class, probability = logreg.classify_germination(soil_moisture)
                predicted_label = "Germinated" if predicted_class == 1 else "Not Germinated"

                plot_url = logreg.create_plot(soil_moisture, predicted_class)
            else:
                error = "Soil moisture must be a value between 0 and 100."

    return render_template(
        "logistic_regression/application.html",
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        probability=probability,
        plot_url=plot_url,
        error=error,
        total_records=logreg.total_records
    )


@app.route("/logistic-regression/evaluation")
def logistic_regression_evaluation():
    return render_template(
        "logistic_regression/evaluation.html",
        **logreg.evaluation_context()
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
                predicted_class, probability = gbc.classify_disease_risk(
                    temperature, humidity, rainfall, nitrogen
                )
                predicted_label = "High Risk" if predicted_class == 1 else "Low Risk"

                plot_url = gbc.create_plot(temperature, humidity, predicted_class)
            else:
                error = "Humidity and Nitrogen must be between 0 and 100, and Rainfall must be 0 or greater."

    return render_template(
        "gradient_boosting/application.html",
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        probability=probability,
        plot_url=plot_url,
        error=error,
        total_records=gbc.total_records
    )


@app.route("/gradient-boosting/evaluation")
def gradient_boosting_evaluation():
    return render_template(
        "gradient_boosting/evaluation.html",
        **gbc.evaluation_context()
    )


# --- Unsupervised Machine Learning ---

@app.route("/unsupervised/concepts")
def unsupervised_concepts():
    return render_template("unsupervised/concepts.html")


@app.route("/unsupervised/manual-exercise")
def unsupervised_manual():
    return render_template(
        "unsupervised/manual.html",
        **unsupervised.manual_exercise_context()
    )


@app.route("/unsupervised/clustering")
def unsupervised_clustering():
    return render_template(
        "unsupervised/clustering.html",
        **unsupervised.clustering_context()
    )


if __name__ == "__main__":
    app.run(debug=True)
