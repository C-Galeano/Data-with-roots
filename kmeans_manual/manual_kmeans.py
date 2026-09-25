"""
Manual K-Means simulation (Part 1 of the Unsupervised Machine Learning module).

Runs 3 iterations of K-Means by hand (no scikit-learn) on data/seed_trays.csv:
  1. Euclidean distance from every tray to the 3 current centroids.
  2. Each tray is assigned to the cluster of its nearest centroid.
  3. New centroids = mean coordinates of the trays assigned to each cluster.

Outputs
  kmeans_manual/results/iteration_<k>.csv  distances + assigned cluster per tray
  kmeans_manual/results/centroids.csv      centroids before/after each iteration
  kmeans_manual/results/variance.csv       within-cluster variance per iteration
  kmeans_manual/results/results.json       everything above, ready for Flask
  static/images/kmeans_manual/*.png        scatter plots and variance chart

Run from the project root with: python kmeans_manual/manual_kmeans.py
"""

import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

DATA_PATH = "data/seed_trays.csv"
RESULTS_DIR = "kmeans_manual/results"
IMAGES_DIR = "static/images/kmeans_manual"

ITERATIONS = 3
INITIAL_CENTROIDS = [
    (30.0, 60.0),   # C1
    (55.0, 45.0),   # C2
    (80.0, 80.0),   # C3
]

X_LABEL = "Soil moisture (%)"
Y_LABEL = "Germination rate (%)"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK = "#52514e"
GRID = "#e4e3df"


# ---------------------------------------------------------------- K-Means core

def euclidean(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def load_trays():
    with open(DATA_PATH, encoding="utf-8") as f:
        return [
            (row["Tray_ID"], float(row["Soil_Moisture_Percent"]),
             float(row["Germination_Rate_Percent"]))
            for row in csv.DictReader(f)
        ]


def run_iteration(trays, centroids):
    """Distance + assignment + update step. Returns rows and new centroids."""
    rows = []
    for tray_id, x, y in trays:
        distances = [euclidean((x, y), c) for c in centroids]
        cluster = distances.index(min(distances))
        rows.append({"tray_id": tray_id, "x": x, "y": y,
                     "distances": distances, "cluster": cluster})

    new_centroids = []
    for k in range(len(centroids)):
        members = [(r["x"], r["y"]) for r in rows if r["cluster"] == k]
        if members:
            new_centroids.append((sum(p[0] for p in members) / len(members),
                                  sum(p[1] for p in members) / len(members)))
        else:  # empty cluster keeps its previous centroid
            new_centroids.append(centroids[k])
    return rows, new_centroids


def within_cluster_variance(rows, centroids):
    """Mean squared distance of each cluster's trays to its centroid, plus SSE."""
    stats = []
    for k, c in enumerate(centroids):
        sq = [euclidean((r["x"], r["y"]), c) ** 2 for r in rows if r["cluster"] == k]
        stats.append({"cluster": k, "records": len(sq), "sse": sum(sq),
                      "variance": sum(sq) / len(sq) if sq else 0.0})
    return stats


# ---------------------------------------------------------------------- plots

def style_axes(ax, title):
    ax.set_title(title, loc="left", fontsize=12, color="#0b0b0b")
    ax.set_xlabel(X_LABEL, color=INK)
    ax.set_ylabel(Y_LABEL, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_initial(trays, centroids, path):
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.scatter([t[1] for t in trays], [t[2] for t in trays], s=36,
               color="#8a8984", edgecolors="white", linewidths=0.8,
               label="Seed trays (unassigned)")
    for k, (cx, cy) in enumerate(centroids):
        ax.scatter(cx, cy, s=260, marker="X", color=COLORS[k],
                   edgecolors="#0b0b0b", linewidths=1.2,
                   label=f"Initial C{k + 1} ({cx:.0f}, {cy:.0f})", zorder=3)
    style_axes(ax, "Seed trays and initial centroids")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_iteration(rows, old_centroids, new_centroids, iteration, path):
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for k in range(len(new_centroids)):
        pts = [(r["x"], r["y"]) for r in rows if r["cluster"] == k]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=36,
                   marker=MARKERS[k], color=COLORS[k], edgecolors="white",
                   linewidths=0.8, label=f"Cluster {k + 1} ({len(pts)} trays)")
    for k, ((ox, oy), (nx, ny)) in enumerate(zip(old_centroids, new_centroids)):
        ax.scatter(ox, oy, s=160, marker="X", facecolors="none",
                   edgecolors=COLORS[k], linewidths=1.2, zorder=3)
        ax.annotate("", xy=(nx, ny), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1))
        ax.scatter(nx, ny, s=260, marker="X", color=COLORS[k],
                   edgecolors="#0b0b0b", linewidths=1.2, zorder=4)
        ax.annotate(f"C{k + 1} ({nx:.1f}, {ny:.1f})", (nx, ny),
                    xytext=(14, -18), textcoords="offset points",
                    fontsize=9, color="#0b0b0b", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID))
    style_axes(ax, f"Iteration {iteration}: clusters and updated centroids")
    ax.scatter([], [], s=120, marker="X", color="#8a8984",
               edgecolors="#0b0b0b", label="Updated centroid")
    ax.scatter([], [], s=100, marker="X", facecolors="none",
               edgecolors="#8a8984", label="Previous centroid")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variance(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    its = [h["iteration"] for h in history]
    wcss = [h["wcss"] for h in history]
    ax.plot(its, wcss, color=COLORS[0], linewidth=2, marker="o", markersize=8)
    for i, w in zip(its, wcss):
        ax.annotate(f"{w:,.1f}", (i, w), xytext=(10, 10),
                    textcoords="offset points", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title("Within-cluster sum of squares (WCSS) by iteration",
                 loc="left", fontsize=12, color="#0b0b0b")
    ax.set_xlabel("Iteration", color=INK)
    ax.set_ylabel("WCSS", color=INK)
    ax.set_xticks(its)
    ax.set_xlim(0.8, 3.4)
    ax.set_ylim(0, max(wcss) * 1.2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    trays = load_trays()
    centroids = list(INITIAL_CENTROIDS)
    plot_initial(trays, centroids, f"{IMAGES_DIR}/initial_centroids.png")

    centroid_rows = [{"iteration": 0, "cluster": k, "x": c[0], "y": c[1], "records": None}
                     for k, c in enumerate(centroids)]
    history = []
    previous = None

    for it in range(1, ITERATIONS + 1):
        rows, new_centroids = run_iteration(trays, centroids)
        stats = within_cluster_variance(rows, new_centroids)
        changed = (None if previous is None else
                   sum(a["cluster"] != b["cluster"] for a, b in zip(rows, previous)))

        with open(f"{RESULTS_DIR}/iteration_{it}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Tray_ID", "Soil_Moisture_Percent", "Germination_Rate_Percent",
                        "Dist_C1", "Dist_C2", "Dist_C3", "Assigned_Cluster"])
            for r in rows:
                w.writerow([r["tray_id"], r["x"], r["y"],
                            *[f"{d:.2f}" for d in r["distances"]], f"C{r['cluster'] + 1}"])

        for k, c in enumerate(new_centroids):
            centroid_rows.append({"iteration": it, "cluster": k, "x": c[0], "y": c[1],
                                  "records": stats[k]["records"]})

        plot_iteration(rows, centroids, new_centroids, it,
                       f"{IMAGES_DIR}/iteration_{it}.png")

        history.append({
            "iteration": it,
            "centroids_before": [list(c) for c in centroids],
            "centroids_after": [list(c) for c in new_centroids],
            "reassigned": changed,
            "clusters": stats,
            "wcss": sum(s["sse"] for s in stats),
            "rows": [{"tray_id": r["tray_id"], "x": r["x"], "y": r["y"],
                      "distances": [round(d, 2) for d in r["distances"]],
                      "cluster": r["cluster"] + 1} for r in rows],
        })
        previous, centroids = rows, new_centroids

    plot_variance(history, f"{IMAGES_DIR}/variance_comparison.png")

    with open(f"{RESULTS_DIR}/centroids.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Iteration", "Cluster", "Soil_Moisture_Percent",
                    "Germination_Rate_Percent", "Records"])
        for c in centroid_rows:
            w.writerow([c["iteration"], f"C{c['cluster'] + 1}", f"{c['x']:.2f}",
                        f"{c['y']:.2f}", "" if c["records"] is None else c["records"]])

    with open(f"{RESULTS_DIR}/variance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Iteration", "Cluster", "Records", "SSE", "Variance", "Reassigned_Trays"])
        for h in history:
            for s in h["clusters"]:
                w.writerow([h["iteration"], f"C{s['cluster'] + 1}", s["records"],
                            f"{s['sse']:.2f}", f"{s['variance']:.2f}",
                            "" if h["reassigned"] is None else h["reassigned"]])
            w.writerow([h["iteration"], "Total (WCSS)", sum(s["records"] for s in h["clusters"]),
                        f"{h['wcss']:.2f}", "", ""])

    with open(f"{RESULTS_DIR}/results.json", "w", encoding="utf-8") as f:
        json.dump({"initial_centroids": [list(c) for c in INITIAL_CENTROIDS],
                   "iterations": history}, f, indent=2)

    for h in history:
        print(f"Iteration {h['iteration']}: reassigned={h['reassigned']} "
              f"WCSS={h['wcss']:.2f}")
        for k, (c, s) in enumerate(zip(h["centroids_after"], h["clusters"])):
            print(f"  C{k + 1}: ({c[0]:.2f}, {c[1]:.2f}) n={s['records']} "
                  f"var={s['variance']:.2f}")


if __name__ == "__main__":
    main()
