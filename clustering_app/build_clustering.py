"""
Clustering Application (Part 2 of the Unsupervised Machine Learning module).

Runs K-Means with scikit-learn on data/crop_recommendation.csv (2,200 records,
7 numerical soil/climate variables) to discover natural farming-condition
profiles, independent of the crop label.

Steps:
  1. Load and clean the dataset (check nulls/duplicates).
  2. Standardize the 7 numerical variables (StandardScaler).
  3. Elbow method to justify the number of clusters k.
  4. Fit KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).
  5. Evaluate with the silhouette score.
  6. Project to 2D with PCA for the scatter plot (centroids projected too).
  7. Save a summary table, cluster profiles and a sample of labeled records.

Outputs
  clustering_app/results/results.json        everything below, ready for Flask
  clustering_app/results/labeled_sample.csv  100-row sample with assigned cluster
  static/images/clustering_app/*.png         elbow, scatter and profile plots

Run from the project root with: python clustering_app/build_clustering.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_recommendation.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "clustering_app", "results")
IMG_DIR = os.path.join(BASE_DIR, "static", "images", "clustering_app")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FEATURE_LABELS = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
    "ph": "Soil pH", "rainfall": "Rainfall (mm)",
}
K = 4
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59d0"]

# --- 1. Load and clean -------------------------------------------------
df = pd.read_csv(DATA_PATH)
n_before = len(df)
n_nulls = int(df.isna().sum().sum())
n_dupes = int(df.duplicated().sum())
df = df.dropna().drop_duplicates().reset_index(drop=True)
n_after = len(df)

X = df[FEATURES].values

# --- 2. Standardize ------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Elbow method (k = 2..9) ------------------------------------------
elbow_wcss = []
elbow_ks = list(range(2, 10))
for k in elbow_ks:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X_scaled)
    elbow_wcss.append(km.inertia_)

plt.figure(figsize=(7, 4.5))
plt.plot(elbow_ks, elbow_wcss, marker="o", color="#2a78d6")
plt.axvline(K, color="#eb6834", linestyle="--", label=f"chosen k = {K}")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS (inertia)")
plt.title("Elbow method")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "elbow.png"), dpi=130)
plt.close()

# --- 4. Fit the final model ----------------------------------------------
kmeans = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df["cluster"] = labels + 1  # 1-indexed for the report

# --- 5. Silhouette score ---------------------------------------------------
sil_score = silhouette_score(X_scaled, labels)

# --- 6. PCA projection for the scatter plot -------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans.cluster_centers_)
explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(8, 6))
for k in range(K):
    mask = labels == k
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, alpha=0.5,
                color=COLORS[k], label=f"Cluster {k + 1} (n={mask.sum()})")
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker="X", s=250,
            c="black", edgecolors="white", linewidths=1.5, zorder=5, label="Centroids")
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% of variance)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% of variance)")
plt.title(f"K-Means clusters (k={K}) — PCA projection")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "clusters_pca.png"), dpi=130)
plt.close()

# --- 7. Cluster summary (centroids back in the original scale) -----------
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

clusters_summary = []
for k in range(K):
    members = df[df["cluster"] == k + 1]
    top_crops = members["label"].value_counts().head(3)
    clusters_summary.append({
        "cluster": k + 1,
        "color": COLORS[k],
        "records": int((labels == k).sum()),
        "pct": round(float((labels == k).sum()) / len(df) * 100, 1),
        "centroid": {f: round(float(v), 2) for f, v in zip(FEATURES, centroids_original[k])},
        "top_crops": [{"crop": c, "count": int(n)} for c, n in top_crops.items()],
    })

# A small boxplot per feature to show how clusters differ (profile chart)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    data_per_cluster = [df[df["cluster"] == k + 1][feat].values for k in range(K)]
    bp = axes[i].boxplot(data_per_cluster, patch_artist=True,
                          tick_labels=[f"C{k+1}" for k in range(K)])
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[i].set_title(FEATURE_LABELS[feat], fontsize=10)
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "feature_profiles.png"), dpi=130)
plt.close()

# --- Sample table for the app (first 100 records + their cluster) --------
sample = df.head(100).copy()
sample.insert(0, "row_id", range(1, len(sample) + 1))
sample.to_csv(os.path.join(RESULTS_DIR, "labeled_sample.csv"), index=False)

results = {
    "dataset": {
        "source_url": "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset",
        "n_before": n_before,
        "n_nulls": n_nulls,
        "n_dupes": n_dupes,
        "n_after": n_after,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "n_crop_labels": int(df["label"].nunique()),
    },
    "model": {
        "k": K,
        "init": "k-means++",
        "n_init": 10,
        "random_state": 42,
    },
    "elbow": {"ks": elbow_ks, "wcss": [round(w, 1) for w in elbow_wcss]},
    "silhouette_score": round(float(sil_score), 4),
    "pca_explained_variance": [round(float(v) * 100, 1) for v in explained_var],
    "clusters": clusters_summary,
    "sample_records": sample.to_dict(orient="records"),
}

with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"Done. k={K}, silhouette={sil_score:.4f}")
for c in clusters_summary:
    print(f"  Cluster {c['cluster']}: {c['records']} records ({c['pct']}%) — top crops: "
          f"{', '.join(t['crop'] for t in c['top_crops'])}")
