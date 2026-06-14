# Iris Flower Clustering Analysis

A self-contained, console-based Python script that performs unsupervised clustering on the classic Iris dataset. The project groups 150 Iris flowers into clusters using K-Means and hierarchical clustering, then evaluates how well those clusters match the real species labels.

> **Reference:** This implementation follows the requirements outlined in [`System_overview.pdf`]([System_overview.pdf](System_overview.pdf)).

---

## 1. Problem Statement

The Iris dataset contains 150 flowers measured across four numeric features:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Each flower belongs to one of three real species:

- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

The goal is to determine whether an unsupervised machine-learning algorithm can automatically discover these three groups **without using the species labels during training**. We also want to measure how closely the discovered clusters align with the true species and identify which species are easiest or hardest to separate.

---

## 2. Solution Overview

The project uses two clustering algorithms and two evaluation metrics:

| Step | Technique | Purpose |
|------|-----------|---------|
| Preprocessing | `StandardScaler` | Put all four features on the same scale (mean ≈ 0, std ≈ 1) so no feature dominates because of its unit size. |
| Clustering 1 | K-Means (`k=3`) | Partition flowers into 3 groups by minimizing within-cluster variance. |
| Clustering 2 | Ward hierarchical clustering | Build a tree (dendrogram) of merges and show how clusters form at different distance thresholds. |
| Evaluation 1 | Silhouette score | Measure how well-separated and compact the K-Means clusters are. |
| Evaluation 2 | Adjusted Rand Index | Compare K-Means cluster assignments to the true species labels. |
| Visualization | Matplotlib (non-interactive `Agg` backend) | Save a scatter plot and a dendrogram as PNG files. |

The entire workflow is implemented in a single file: [`iris_clustering_analysis.py`](./iris_clustering_analysis.py).

---

## 3. How the Data Was Processed

The script executes the following pipeline:

1. **Load & validate data**
   - Loads the Iris dataset via `sklearn.datasets.load_iris`.
   - Confirms exactly **150 samples** and **4 features**.
   - Checks for missing/null values.
   - Exits with a clear `ERROR:` message if any validation fails.

2. **Standardize features**
   - Applies `StandardScaler` to the four numeric features.
   - Wraps the operation in error handling and exits with `ERROR: Standardization failed` if it fails.

3. **K-Means clustering**
   - Runs `KMeans(n_clusters=3, random_state=42, n_init=10)` on standardized data.
   - Prints cluster sizes and cluster centers.

4. **Hierarchical clustering**
   - Builds a linkage matrix with `scipy.cluster.hierarchy.linkage(method='ward', metric='euclidean')`.
   - Generates a dendrogram with a horizontal cut line at the height that produces 3 clusters.

5. **Evaluation**
   - Computes the silhouette score for the K-Means result.
   - Computes the adjusted Rand index against the true species labels.

6. **Visualization**
   - Saves `kmeans_scatter.png`: sepal length vs. sepal width, colored by K-Means cluster.
   - Saves `dendrogram.png`: full Ward dendrogram with a 3-cluster cut line.

7. **Console report**
   - Prints a numbered `[1]`–`[6]` progress report with checkmarks.
   - Prints a final `SUCCESS:` summary block.
   - Prints an `ANALYSIS` section interpreting the scores and identifying the best/worst distinguished species from the cluster-vs-label cross-tabulation.

---

## 4. Results

A sample run produces output similar to:

```text
[1] Loading Iris dataset...
    ✓ Loaded 150 samples with 4 features
    ✓ No missing values detected

[2] Preprocessing: Standardizing features...
    ✓ Features standardized (mean ~0, std ~1)

[3] Running K-Means clustering...
    ✓ K-Means converged
    Cluster sizes:
      Cluster 0: 53 samples
      Cluster 1: 50 samples
      Cluster 2: 47 samples
    ...

[5] Evaluating clustering performance...
    ✓ Silhouette Score: 0.4599
    ✓ Adjusted Rand Index: 0.6201
    ...

SUCCESS: All tasks completed
  Samples loaded:        150
  Features loaded:       4
  Standardization:       Done
  K-Means cluster count: 3
  Hierarchical status:   Done
  Silhouette score:      0.4599
  Rand index:            0.6201
  Saved files:           kmeans_scatter.png, dendrogram.png
```

### Interpretation

- **Silhouette score ≈ 0.46** → The clusters are **moderately separated** (between 0.25 and 0.5).
- **Adjusted Rand Index ≈ 0.62** → The K-Means grouping shows **moderate agreement** with the real species labels (between 0.5 and 0.7).
- **Species distinction:**
  - *Setosa* is usually distinguished perfectly (100% in one cluster) because it is linearly separable.
  - *Versicolor* and *Virginica* overlap in feature space, causing most of the confusion.
  - *Virginica* is typically the worst distinguished of the three.

Two PNG images are saved in the project root:

- `kmeans_scatter.png`
- `dendrogram.png`

---

## 5. How to Run the Project

### Prerequisites

- Python **3.12+**
- Dependencies: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`

### Option A: Using the provided virtual environment

```bash
cd /home/iron/dev/Projects/Iris_clustering_analysis
source .venv/bin/activate
python iris_clustering_analysis.py
```

### Option B: Using `uv` / `pip`

If you prefer to install dependencies from `pyproject.toml`:

```bash
# With uv
uv run iris_clustering_analysis.py

# Or with pip in a fresh environment
pip install -e .
python iris_clustering_analysis.py
```

### Expected outputs

After running, the console displays the full numbered report, the `SUCCESS:` block, and the `ANALYSIS` section. The following files are generated in the project directory:

- `kmeans_scatter.png`
- `dendrogram.png`

No GUI, user input, or database is required.

---

## 6. Project Files

| File | Description |
|------|-------------|
| `iris_clustering_analysis.py` | Main clustering script (single self-contained file). |
| `pyproject.toml` | Project metadata and Python dependencies. |
| `SYSTEM OVERVIEW. data mining.pdf` | Assignment/system overview reference document. |
| `kmeans_scatter.png` | Generated scatter plot (created at runtime). |
| `dendrogram.png` | Generated dendrogram (created at runtime). |

---

## 7. Error Handling

The script validates each stage and exits with a specific error message on failure:

| Scenario | Error message |
|----------|---------------|
| Dataset fails to load | `ERROR: Could not load Iris dataset` |
| Wrong sample count | `ERROR: Dataset has incorrect number of samples` |
| Wrong feature count | `ERROR: Dataset has incorrect number of features` |
| Missing/null values | `ERROR: Dataset contains missing values` |
| Standardization fails | `ERROR: Standardization failed` |
| K-Means fails | `ERROR: K-Means failed to converge` |
| Hierarchical clustering fails | `ERROR: Hierarchical clustering failed` |
| True labels unavailable | `ERROR: Cannot calculate Rand Index - real labels missing` |
