"""
Unsupervised clustering analysis on the Iris dataset.

Loads the Iris data, validates it, standardizes features, runs K-Means and
hierarchical (Ward) clustering, evaluates the results, and saves two PNG
visualizations.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def load_and_validate_iris():
    """Load Iris dataset and validate shape / missing values."""
    print("[1] Loading Iris dataset...")
    try:
        iris = load_iris()
    except Exception:
        print("ERROR: Could not load Iris dataset")
        sys.exit(1)

    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    if X.shape[0] != 150:
        print("ERROR: Dataset has incorrect number of samples")
        sys.exit(1)
    if X.shape[1] != 4:
        print("ERROR: Dataset has incorrect number of features")
        sys.exit(1)
    if np.isnan(X).any():
        print("ERROR: Dataset contains missing values")
        sys.exit(1)

    print("    \u2713 Loaded 150 samples with 4 features")
    print("    \u2713 No missing values detected")
    return X, y, feature_names, target_names


def standardize_features(X):
    """Standardize features to zero mean and unit variance."""
    print("\n[2] Preprocessing: Standardizing features...")
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    except Exception:
        print("ERROR: Standardization failed")
        sys.exit(1)

    print("    \u2713 Features standardized (mean ~0, std ~1)")
    return X_scaled


def run_kmeans(X_scaled):
    """Run K-Means clustering with k=3."""
    print("\n[3] Running K-Means clustering...")
    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
    except Exception:
        print("ERROR: K-Means failed to converge")
        sys.exit(1)

    cluster_sizes = pd.Series(kmeans_labels).value_counts().sort_index()
    print("    \u2713 K-Means converged")
    print("    Cluster sizes:")
    for cluster_id, count in cluster_sizes.items():
        print(f"      Cluster {cluster_id}: {count} samples")
    print("    Cluster centers (standardized space):")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"      Cluster {i}: {center}")

    return kmeans_labels, kmeans.cluster_centers_


def run_hierarchical(X_scaled):
    """Run hierarchical clustering using Ward's method."""
    print("\n[4] Running Hierarchical clustering (Ward linkage)...")
    try:
        linkage_matrix = linkage(X_scaled, method="ward", metric="euclidean")
    except Exception:
        print("ERROR: Hierarchical clustering failed")
        sys.exit(1)

    print("    \u2713 Hierarchical clustering completed")
    return linkage_matrix


def evaluate_clustering(X_scaled, true_labels, kmeans_labels, target_names):
    """Compute silhouette score and adjusted Rand index."""
    print("\n[5] Evaluating clustering performance...")

    try:
        silhouette = silhouette_score(X_scaled, kmeans_labels)
    except Exception:
        silhouette = None

    if true_labels is not None and len(true_labels) == len(kmeans_labels):
        rand_index = adjusted_rand_score(true_labels, kmeans_labels)
        labels_available = True
    else:
        rand_index = None
        labels_available = False
        print("ERROR: Cannot calculate Rand Index - real labels missing")

    if silhouette is not None:
        print(f"    \u2713 Silhouette Score: {silhouette:.4f}")
    if labels_available:
        print(f"    \u2713 Adjusted Rand Index: {rand_index:.4f}")

    # Cross-tabulation for species/cluster agreement
    cross_tab = pd.crosstab(
        pd.Series(true_labels, name="Species").map(
            {i: name for i, name in enumerate(target_names)}
        ),
        pd.Series(kmeans_labels, name="K-Means Cluster"),
    )
    print("\n    Species vs K-Means Cluster cross-tabulation:")
    print(cross_tab.to_string())

    return silhouette, rand_index, cross_tab


def visualize(
    X,
    feature_names,
    kmeans_labels,
    linkage_matrix,
    silhouette,
    rand_index,
    target_names,
):
    """Save K-Means scatter plot and dendrogram as PNG files."""
    print("\n[6] Generating visualizations...")

    # K-Means scatter plot: sepal length vs sepal width
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=kmeans_labels,
        cmap="viridis",
        edgecolor="k",
        s=70,
        alpha=0.8,
    )
    ax.set_title("K-Means Clustering: Sepal Length vs Sepal Width")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="upper right",
    )
    ax.add_artist(legend)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")
    plt.tight_layout()
    plt.savefig("kmeans_scatter.png", dpi=150)
    plt.close(fig)
    print("    \u2713 Saved kmeans_scatter.png")

    # Dendrogram with horizontal cut line at height yielding 3 clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode="lastp",
        p=150,
        leaf_rotation=90.0,
        leaf_font_size=6.0,
        show_leaf_counts=True,
    )

    # Determine the height that yields exactly 3 clusters
    heights = linkage_matrix[:, 2]
    sorted_heights = np.sort(heights)
    # Number of clusters starts at N and decreases by 1 at each merge.
    # The merge that reduces from 4 to 3 clusters is at index (N - 4).
    # A horizontal cut at that height therefore yields 3 clusters.
    cut_height = sorted_heights[X.shape[0] - 4]

    ax.axhline(y=cut_height, color="r", linestyle="--", label="cut for 3 clusters")
    ax.set_title("Hierarchical Clustering Dendrogram (Ward)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Distance")
    ax.legend()
    plt.tight_layout()
    plt.savefig("dendrogram.png", dpi=150)
    plt.close(fig)
    print("    \u2713 Saved dendrogram.png")


def print_summary(
    X,
    kmeans_labels,
    silhouette,
    rand_index,
    standardization_success=True,
    hierarchical_success=True,
):
    """Print final structured success summary block."""
    print("\n" + "=" * 60)
    print("SUCCESS: All tasks completed")
    print("=" * 60)
    print(f"  Samples loaded:        {X.shape[0]}")
    print(f"  Features loaded:       {X.shape[1]}")
    print(f"  Standardization:       {'Done' if standardization_success else 'Failed'}")
    print(f"  K-Means cluster count: {len(np.unique(kmeans_labels))}")
    print(
        f"  Hierarchical status:   {'Done' if hierarchical_success else 'Failed'}"
    )
    if silhouette is not None:
        print(f"  Silhouette score:      {silhouette:.4f}")
    else:
        print("  Silhouette score:      N/A")
    if rand_index is not None:
        print(f"  Rand index:            {rand_index:.4f}")
    else:
        print("  Rand index:            N/A")
    print("  Saved files:           kmeans_scatter.png, dendrogram.png")
    print("=" * 60)


def print_analysis(silhouette, rand_index, cross_tab, target_names):
    """Print plain-English interpretation of the clustering results."""
    print("\nANALYSIS")
    print("-" * 60)

    if silhouette is None:
        silhouette_text = "Silhouette score could not be calculated."
    elif silhouette >= 0.5:
        silhouette_text = (
            f"Silhouette score = {silhouette:.4f}: the clusters are well-separated."
        )
    elif silhouette >= 0.25:
        silhouette_text = (
            f"Silhouette score = {silhouette:.4f}: the clusters are moderately separated."
        )
    else:
        silhouette_text = (
            f"Silhouette score = {silhouette:.4f}: the clusters show poor separation."
        )
    print(silhouette_text)

    if rand_index is None:
        rand_text = "Adjusted Rand index could not be calculated because true labels were unavailable."
    elif rand_index >= 0.7:
        rand_text = (
            f"Adjusted Rand index = {rand_index:.4f}: strong agreement with true species labels."
        )
    elif rand_index >= 0.5:
        rand_text = (
            f"Adjusted Rand index = {rand_index:.4f}: moderate agreement with true species labels."
        )
    else:
        rand_text = (
            f"Adjusted Rand index = {rand_index:.4f}: poor agreement with true species labels."
        )
    print(rand_text)

    # Identify best/worst distinguished species from the cross-tabulation.
    # "Best distinguished" = highest proportion of that species assigned to a single cluster.
    species_dominance = {}
    for species in cross_tab.index:
        row = cross_tab.loc[species]
        species_dominance[species] = row.max() / row.sum()

    best_species = max(species_dominance, key=species_dominance.get)
    worst_species = min(species_dominance, key=species_dominance.get)

    print(
        f"\nBest distinguished species: {best_species} "
        f"({species_dominance[best_species]:.1%} of its samples assigned to one K-Means cluster)."
    )
    print(
        f"Worst distinguished species: {worst_species} "
        f"({species_dominance[worst_species]:.1%} of its samples assigned to one K-Means cluster)."
    )
    print(
        "Note: Setosa is usually the easiest to separate because it is linearly "
        "separable in feature space, while Versicolor and Virginica often overlap."
    )
    print("-" * 60)


def main():
    X, y, feature_names, target_names = load_and_validate_iris()
    X_scaled = standardize_features(X)
    kmeans_labels, _ = run_kmeans(X_scaled)
    linkage_matrix = run_hierarchical(X_scaled)
    silhouette, rand_index, cross_tab = evaluate_clustering(
        X_scaled, y, kmeans_labels, target_names
    )
    visualize(
        X,
        feature_names,
        kmeans_labels,
        linkage_matrix,
        silhouette,
        rand_index,
        target_names,
    )
    print_summary(X, kmeans_labels, silhouette, rand_index)
    print_analysis(silhouette, rand_index, cross_tab, target_names)


if __name__ == "__main__":
    main()
