# Data with Roots

University Machine Learning project built as a Flask web application. It presents Machine
Learning concepts, real-world use cases, and guided walkthroughs of Linear Regression and
Logistic Regression classification.

## Objective

Provide a clear, well-structured web app that explains Machine Learning fundamentals and
showcases supervised learning algorithms (Linear Regression and Logistic Regression),
combining an academic presentation with a clean, modern design.

## Technologies

- Python
- Flask
- Pandas / scikit-learn
- HTML5 + Jinja2
- Bootstrap 5
- Git / GitHub

## Project Structure

```text
data-with-roots/
│
├── app.py                 # Flask app: routes only
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/                # One module per ML section (data loading, training, plots)
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── gradient_boosting.py
│   └── unsupervised.py    # Loads the K-Means results for the Unsupervised pages
│
├── utils/                 # Helpers shared by the model modules
│   ├── paths.py           # Absolute project paths (BASE_DIR, DATA_DIR)
│   ├── plots.py           # fig_to_base64(), confusion matrix plot
│   └── metrics.py         # Accuracy, precision, recall, F1, confusion matrix
│
├── kmeans_manual/         # Unsupervised > Part 1: manual K-Means simulation
│   ├── manual_kmeans.py
│   ├── README.md          # Write-up of the manual exercise
│   └── results/           # Iteration tables, centroids, variance, results.json
│
├── clustering_app/        # Unsupervised > Clustering Application (scikit-learn K-Means)
│   ├── build_clustering.py
│   └── results/           # results.json, labeled sample
│
├── data/
│   ├── coffee_shop_revenue.csv                 # Linear Regression dataset
│   ├── seed_germination.csv                    # Logistic Regression dataset
│   ├── generate_seed_germination_dataset.py
│   ├── crop_disease_risk.csv                   # Gradient Boosting dataset
│   ├── generate_crop_disease_dataset.py
│   ├── seed_trays.csv                          # Manual K-Means dataset (100 records)
│   ├── generate_seed_trays_dataset.py
│   └── crop_recommendation.csv                 # Clustering Application dataset
│
├── templates/
│   ├── base.html          # Shared layout: navbar, footer, blocks
│   ├── _macros.html
│   ├── home.html          # Landing page
│   ├── ml/                # Machine Learning: concepts, types
│   ├── use_cases/         # Use Cases 1-4
│   ├── regression/        # Supervised > Linear Regression: concepts, application
│   ├── logistic_regression/  # Supervised > Logistic Regression: concepts, application, evaluation
│   ├── gradient_boosting/    # Supervised > Gradient Boosting: concepts, application, evaluation
│   └── unsupervised/      # Unsupervised: concepts, manual exercise, clustering
│
└── static/
    ├── css/style.css
    ├── js/script.js
    └── images/
        ├── kmeans_manual/     # Plots of the manual K-Means iterations
        └── clustering_app/    # Elbow, PCA scatter and cluster profile plots
```

To add a new section: put its data loading, training and plotting in a new file under
`models/`, and add only the routes to `app.py`.

## Getting Started

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   - Windows: `venv\Scripts\activate`
   - macOS / Linux: `source venv/bin/activate`

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python app.py
   ```

4. Open your browser at `http://127.0.0.1:5000`.

## Routes

| Route                          | Description                       |
|---------------------------------|------------------------------------|
| `/`                              | Home                               |
| `/ml/concepts`                   | Machine Learning Concepts          |
| `/ml/types`                      | Types of Machine Learning          |
| `/use-cases/1..4`                | Use Cases 1 to 4                   |
| `/regression/concepts`           | Linear Regression - Concepts       |
| `/regression/application`        | Linear Regression - Application    |
| `/logistic-regression/concepts`     | Logistic Regression - Concepts     |
| `/logistic-regression/application`  | Logistic Regression - Application  |
| `/logistic-regression/evaluation`   | Logistic Regression - Evaluation   |
| `/gradient-boosting/concepts`       | Gradient Boosting - Concepts       |
| `/gradient-boosting/application`    | Gradient Boosting - Application    |
| `/gradient-boosting/evaluation`     | Gradient Boosting - Evaluation     |
| `/unsupervised/concepts`           | Unsupervised Machine Learning - Concepts |
| `/unsupervised/manual-exercise`    | Unsupervised Machine Learning - Manual Exercise |
| `/unsupervised/clustering`         | Unsupervised Machine Learning - Clustering Application |

## Status

Activity 1 (base structure, navigation, Machine Learning, Use Cases and Linear Regression)
is complete. Activity 2 adds Logistic Regression (Concepts + Application) for the Seed
Germination Prediction problem. Evaluation Metrics for Logistic Regression and the group's
assigned classification model (Gradient Boosting Classifier) are being developed by the
rest of the team on this same `R1A2` branch.
