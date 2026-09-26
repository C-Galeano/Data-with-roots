"""
Supervised: Linear Regression.

Topic: Coffee shop revenue.
Independent variable: Marketing_Spend_Per_Day
Target variable     : Number_of_Customers_Per_Day
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from utils.paths import DATA_DIR
from utils.plots import fig_to_base64

df = pd.read_csv(os.path.join(DATA_DIR, "coffee_shop_revenue.csv"))

x = df[["Marketing_Spend_Per_Day"]]
y = df[["Number_of_Customers_Per_Day"]]

total_records = len(df)

model = LinearRegression()
model.fit(x, y)


def calculate_customers(marketing_spend):
    input_df = pd.DataFrame({"Marketing_Spend_Per_Day": [marketing_spend]})
    result = model.predict(input_df)[0][0]
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
        label="Actual data"
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
        label="Regression line"
    )

    if marketing_spend is not None and predicted_customers is not None:
        plt.scatter(
            marketing_spend,
            predicted_customers,
            s=180,
            color="black",
            marker="X",
            zorder=5,
            label="Your prediction"
        )

    plt.xlabel("Marketing Spend Per Day")
    plt.ylabel("Number of Customers Per Day")
    plt.title("Linear Regression: Marketing Spend vs Number of Customers")
    plt.legend()
    plt.grid(alpha=0.3)

    return fig_to_base64()
