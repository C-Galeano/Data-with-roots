"""Plot helpers shared by every model module."""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_to_base64():
    """Save the current matplotlib figure as a base64 PNG and close it."""
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()

    return plot_url


def create_confusion_matrix_plot(cm, classes, title):
    """Confusion matrix heatmap with the count written in each cell."""
    plt.figure(figsize=(7, 5))

    plt.imshow(cm, interpolation="nearest", cmap="Blues")

    plt.title(title)
    plt.colorbar()

    plt.xticks([0, 1], classes, rotation=20)
    plt.yticks([0, 1], classes)

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=14,
                fontweight="bold"
            )

    plt.tight_layout()

    return fig_to_base64()
