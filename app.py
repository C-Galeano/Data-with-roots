from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64


from sklearn.linear_model import LinearRegression

app = Flask(__name__)

df = pd.read_csv("data/coffee_shop_revenue.csv")

x = df[["Marketing_Spend_Per_Day"]]
y = df[["Number_of_Customers_Per_Day"]]

total_records = len(df)

model = LinearRegression()
model.fit(x, y)


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
if __name__ == "__main__":
    app.run(debug=True)
