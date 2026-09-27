"""
Unsupervised Machine Learning: K-Means.

The heavy work runs offline and saves its results as JSON:
  - Manual Exercise        : kmeans_manual/manual_kmeans.py      -> kmeans_manual/results/results.json
  - Clustering Application : clustering_app/build_clustering.py  -> clustering_app/results/results.json
This module only loads those results and prepares them for the templates.
"""

import json
import os

from utils.paths import BASE_DIR

MANUAL_CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # same colors as the plots

with open(os.path.join(BASE_DIR, "kmeans_manual", "results", "results.json")) as f:
    manual_kmeans = json.load(f)

with open(os.path.join(BASE_DIR, "clustering_app", "results", "results.json")) as f:
    clustering_results = json.load(f)


def manual_exercise_context():
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


def clustering_context():
    """Results of the scikit-learn K-Means run, as saved by build_clustering.py."""
    return clustering_results
