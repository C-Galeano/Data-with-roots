from flask import Flask, render_template

app = Flask(__name__)


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


@app.route("/regression/application")
def regression_application():
    return render_template("regression/application.html")


if __name__ == "__main__":
    app.run(debug=True)
