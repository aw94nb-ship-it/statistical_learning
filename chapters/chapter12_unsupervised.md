# Chapter 12: Unsupervised Learning

Unsupervised learning finds structure in data without a response variable Y. The two main methods are **PCA** (dimension reduction) and **clustering** (grouping observations into similar clusters).

---

## Key Concepts

### Principal Components Analysis (PCA)

PCA finds low-dimensional representations of high-dimensional data by identifying **directions of maximum variance**.

**First principal component** $Z_1$: the linear combination of features with maximum variance:

$$Z_1 = \phi_{11} X_1 + \phi_{21} X_2 + \cdots + \phi_{p1} X_p$$

subject to $\sum_{j=1}^p \phi_{j1}^2 = 1$ (unit norm). The vector $\phi_1 = (\phi_{11}, \ldots, \phi_{p1})$ is the first **loading vector** — the direction in p-dimensional space with the most variance.

**Second principal component** $Z_2$: maximum variance direction **orthogonal to $Z_1$**, and so on.

The **scores** $z_{i1} = X_i^T \phi_1$ are the projections of observation $i$ onto the first PC.

**Variance explained**: the **proportion of variance explained (PVE)** by PC $m$:

$$\text{PVE}_m = \frac{\sum_{i=1}^n z_{im}^2}{\sum_{j=1}^p \sum_{i=1}^n x_{ij}^2}$$

Total PVE of first M components: $\sum_{m=1}^M \text{PVE}_m$. Plot the **scree plot** to choose M — look for the "elbow" where the curve flattens.

**Must center (and usually scale) variables before PCA**: without centering, the first PC reflects means, not variance. Without scaling, high-variance variables dominate.

**Interpretation of loadings**:
- Large $|\phi_{jm}|$: variable j is strongly associated with PC m
- Sign: observations with large positive scores on PC m have large values of variables with positive loadings (and small values of variables with negative loadings)

**Biplot**: shows both observation scores and variable loadings simultaneously — variables pointing in similar directions are correlated; observation position shows their values on the PCs.

---

### How Many PCs to Use?

1. **Scree plot elbow**: choose M where PVE stops dropping sharply
2. **Cumulative PVE threshold**: keep enough PCs to explain 70–90% of variance
3. **Cross-validation**: if PCA is a preprocessing step, choose M to minimize CV error of downstream model
4. **Kaiser criterion**: keep PCs with eigenvalues > 1 (variance above average)

No single right answer — context-dependent.

---

### K-Means Clustering

Partitions n observations into K non-overlapping clusters to minimize **within-cluster variation**:

$$\min_{C_1, \ldots, C_K} \sum_{k=1}^K W(C_k), \quad W(C_k) = \frac{1}{|C_k|} \sum_{i,i' \in C_k} \sum_{j=1}^p (x_{ij} - x_{i'j})^2$$

Equivalently: minimize the total within-cluster sum of squares (WCSS).

**Algorithm**:
1. Randomly assign each observation to one of K clusters
2. Compute cluster centroids (mean of each cluster)
3. Reassign each observation to the nearest centroid
4. Repeat steps 2–3 until assignments stop changing

**Guaranteed to converge** (each step reduces WCSS) but **not to the global minimum** (depends on initialization). Always run with multiple random initializations and keep the best.

**Choosing K**:
- **Elbow method**: plot WCSS vs. K; choose K at the "elbow" (diminishing returns)
- **Silhouette score**: measures how similar an observation is to its own cluster vs. neighboring clusters (range −1 to 1, higher is better)
- **Gap statistic**: compare WCSS to expected WCSS under a null reference distribution
- **Domain knowledge**: often the most reliable

**Limitations**:
- Assumes spherical, equally-sized clusters (Euclidean distance + centroid structure)
- Sensitive to outliers (outliers pull centroids)
- Must specify K in advance
- Results depend on scale — standardize first

---

### Hierarchical Clustering

Builds a **dendrogram** — a tree diagram showing all possible clusterings simultaneously, without specifying K in advance.

**Agglomerative (bottom-up) algorithm**:
1. Start: each observation is its own cluster (n clusters)
2. Compute pairwise dissimilarities between all clusters
3. Merge the two most similar clusters
4. Repeat until all observations are in one cluster

Read the dendrogram top-to-bottom: horizontal cuts at different heights give different numbers of clusters. The **height of a merge** represents the dissimilarity between the merged clusters.

**Linkage methods** — how dissimilarity between clusters is measured:

| Linkage | Distance between clusters | Properties |
|---|---|---|
| **Complete** | Max distance between any pair | Balanced, compact clusters |
| **Single** | Min distance between any pair | Can produce "chaining" |
| **Average** | Average distance between all pairs | Compromise |
| **Centroid** | Distance between centroids | Can produce inversions |
| **Ward** | Increase in WCSS from merging | Tends to produce equal-size clusters |

**Choice of dissimilarity measure**:
- **Euclidean distance**: default; sensitive to scale → standardize first
- **Correlation-based**: clusters based on pattern, not magnitude — useful when scale doesn't matter (e.g., gene expression)
- **Manhattan distance**: less sensitive to outliers

---

### K-Means vs. Hierarchical Clustering

| | K-Means | Hierarchical |
|---|---|---|
| **Specify K?** | Yes, required | No (choose by cutting dendrogram) |
| **Deterministic?** | No (random init) | Yes (same result every time) |
| **Scalability** | Fast, O(nKp) | Slow, O(n² log n) or O(n³) |
| **Cluster shape** | Spherical | Arbitrary (linkage-dependent) |
| **Output** | Hard assignments | Dendrogram (all granularities) |

---

### Practical Considerations

**Scaling**: both K-means and PCA are sensitive to variable scale. Standardize (zero mean, unit variance) unless variables are on a natural common scale.

**Outliers**: a single outlier can form its own cluster in hierarchical clustering, or pull a centroid in K-means. Consider robust variants or removing outliers first.

**Mixed data**: Euclidean distance doesn't naturally handle categorical variables. Use Gower distance (handles mixed types) or encode categoricals appropriately.

**Validation**: clustering has no ground-truth label to validate against. Use:
- Internal metrics: silhouette score, WCSS
- External metrics (if labels exist): adjusted Rand index, normalized mutual information
- Visual inspection: t-SNE or UMAP for 2D visualization of high-dimensional clusters

---

## Real Data Example: NCI Microarray Data + USArrests

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# ── USArrests: PCA for dimensionality reduction ───────────────────────────────
# USArrests: 50 US states, 4 crime statistics
try:
    from ISLP import load_data
    USArrests = load_data("USArrests")
except ImportError:
    url = "https://raw.githubusercontent.com/rdatasets/rdatasets/main/csv/datasets/USArrests.csv"
    USArrests = pd.read_csv(url, index_col=0)

# Standardize (variables on different scales: Murder 0-18, UrbanPop 32-91)
scaler = StandardScaler()
X_arrests = scaler.fit_transform(USArrests)

# Fit PCA
pca = PCA()
scores = pca.fit_transform(X_arrests)
loadings = pca.components_   # shape: (n_components, n_features)

print("PC loadings:")
loadings_df = pd.DataFrame(loadings.T, index=USArrests.columns,
                             columns=[f"PC{i+1}" for i in range(4)])
print(loadings_df.round(3))

print(f"\nVariance explained:")
for i, pve in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {pve:.3f} ({pca.explained_variance_ratio_[:i+1].sum():.3f} cumulative)")

# Scree plot + biplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].plot(range(1, 5), pca.explained_variance_ratio_ * 100, "o-", color="steelblue")
axes[0].plot(range(1, 5), np.cumsum(pca.explained_variance_ratio_) * 100,
             "s--", color="coral", label="Cumulative")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_title("Scree Plot: USArrests")
axes[0].legend()
axes[0].axhline(100, color="gray", linestyle=":")

# Biplot (PC1 vs PC2)
axes[1].scatter(scores[:, 0], scores[:, 1], alpha=0.5, color="steelblue", s=40)
for i, state in enumerate(USArrests.index):
    axes[1].annotate(state, (scores[i, 0], scores[i, 1]), fontsize=6, alpha=0.7)

scale = 3
for j, feat in enumerate(USArrests.columns):
    axes[1].arrow(0, 0, loadings[0, j]*scale, loadings[1, j]*scale,
                  color="coral", head_width=0.05, linewidth=2)
    axes[1].text(loadings[0, j]*scale*1.1, loadings[1, j]*scale*1.1,
                 feat, color="coral", fontsize=9)

axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].axvline(0, color="black", linewidth=0.5)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("Biplot: PC1 vs PC2")
plt.tight_layout()
plt.show()

# ── K-Means: Choosing K ───────────────────────────────────────────────────────
wcss = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_arrests)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_arrests, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_range), wcss, "o-", color="steelblue")
axes[0].set_xlabel("K")
axes[0].set_ylabel("WCSS (inertia)")
axes[0].set_title("Elbow Method")

axes[1].plot(list(K_range), sil_scores, "o-", color="coral")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score")
plt.tight_layout()
plt.show()

best_K = list(K_range)[np.argmax(sil_scores)]
print(f"\nBest K (silhouette): {best_K}")

km_final = KMeans(n_clusters=best_K, n_init=20, random_state=42)
labels_km = km_final.fit_predict(X_arrests)

# Plot clusters in PCA space
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=labels_km,
                     cmap="tab10", s=60, alpha=0.8)
for i, state in enumerate(USArrests.index):
    ax.annotate(state, (scores[i, 0], scores[i, 1]), fontsize=6, alpha=0.8)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"K-Means (K={best_K}) in PCA Space")
plt.tight_layout()
plt.show()

# ── Hierarchical Clustering ───────────────────────────────────────────────────
# Compute linkage matrix
Z_complete = linkage(X_arrests, method="complete", metric="euclidean")
Z_average  = linkage(X_arrests, method="average",  metric="euclidean")
Z_single   = linkage(X_arrests, method="single",   metric="euclidean")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, Z, method in zip(axes,
                          [Z_complete, Z_average, Z_single],
                          ["Complete", "Average", "Single"]):
    dendrogram(Z, labels=list(USArrests.index), leaf_rotation=90,
               leaf_font_size=7, ax=ax)
    ax.set_title(f"Hierarchical Clustering ({method} Linkage)")
    ax.set_xlabel("State")
    ax.set_ylabel("Dissimilarity")

plt.tight_layout()
plt.show()

# Cut dendrogram at K=4 clusters
from scipy.cluster.hierarchy import fcluster
labels_hc = fcluster(Z_complete, t=4, criterion="maxclust")
print(f"\nHierarchical clusters (K=4, complete linkage):")
for k in range(1, 5):
    states = USArrests.index[labels_hc == k].tolist()
    print(f"  Cluster {k} ({len(states)}): {', '.join(states[:6])}{'...' if len(states)>6 else ''}")

# ── PCA then cluster (common pipeline) ───────────────────────────────────────
pca2 = PCA(n_components=2)
scores_2d = pca2.fit_transform(X_arrests)

km_pca = KMeans(n_clusters=4, n_init=20, random_state=42)
labels_pca_km = km_pca.fit_predict(scores_2d)

print(f"\nCluster sizes (K-means on 2 PCs): {np.bincount(labels_pca_km)}")
```

---

## Interview Questions

**Q1: What is PCA and what problem does it solve?**

PCA finds a low-dimensional representation of high-dimensional data by identifying the directions of maximum variance. The first PC is the direction along which the data has highest variance; each subsequent PC is orthogonal to the previous and captures the next most variance. The scores (projections onto PCs) are the new low-dimensional coordinates. PCA is used for: (1) visualization — projecting to 2–3 dimensions for plotting; (2) preprocessing — reducing p before feeding into a model; (3) noise reduction — low-variance PCs often represent noise. Must standardize first when variables are on different scales.

---

**Q2: How do you interpret PCA loadings and scores?**

**Loadings** ($\phi_{jm}$): the weight of feature j on PC m. Large positive loading → feature j has a large positive contribution to PC m. Variables with similar loading vectors are positively correlated. Variables with opposite loading signs are negatively correlated. **Scores** ($z_{im}$): the projection of observation i onto PC m. A large positive score on PC 1 means the observation is "high" on variables with positive loadings for PC 1. In a biplot: observation position shows their scores; arrow direction/length shows variable loadings — variables pointing right have high values for observations on the right.

---

**Q3: How does K-means work and what can go wrong?**

K-means iterates: assign each observation to the nearest centroid, recompute centroids, repeat. Each step reduces WCSS — it converges but to a local minimum. What can go wrong: (1) **initialization sensitivity** — fix with multiple random restarts (n_init=20+) or K-means++ initialization; (2) **sensitive to outliers** — outliers pull centroids away from the true cluster center; (3) **assumes spherical clusters** — fails for elongated or irregular shapes; (4) **scale sensitivity** — standardize first; (5) **choosing K** — there's no single right answer; use elbow, silhouette, or domain knowledge together.

---

**Q4: What is a dendrogram and how do you use it?**

A dendrogram is a tree diagram produced by hierarchical clustering. It shows all observations as leaves; each internal node represents a merge. The **height of a merge** is the dissimilarity between the two clusters being merged — higher = more different. To get K clusters: draw a horizontal cut through the dendrogram at the appropriate height; the number of branches crossing the line = the number of clusters. The dendrogram is useful because it shows all possible clusterings simultaneously, without committing to a K upfront — you can explore different granularities.

---

**Q5: How does the choice of linkage affect hierarchical clustering?**

Linkage defines how the distance between two clusters is measured. **Complete linkage**: max distance between any pair — produces compact, balanced clusters, least prone to chaining. **Single linkage**: min distance — prone to "chaining" (one long cluster that grows by picking up nearby outliers). **Average linkage**: average pairwise distance — compromise between complete and single. **Ward**: minimizes the increase in WCSS from merging — tends to produce equal-sized, compact clusters (similar to K-means behavior). In practice, Ward and complete linkage are most commonly used.

---

**Q6: When would you use PCA vs. t-SNE for visualization?**

PCA is a linear method — it preserves global structure (distances between well-separated clusters) and is deterministic and fast. t-SNE is non-linear — it preserves local neighborhood structure (nearby points in high-d stay nearby in 2D) but distorts global distances. Use PCA when: you need to understand which variables drive the variation; you need interpretable components; you'll use the low-dimensional space for downstream modeling. Use t-SNE when: the goal is purely 2D visualization; you care more about local cluster structure than global geometry; you have complex non-linear manifold structure (e.g., image embeddings, single-cell RNA-seq). t-SNE is not useful for supervised learning preprocessing — only PCA/UMAP are.

---

**Q7: What is the difference between K-means and hierarchical clustering in practice?**

K-means requires specifying K upfront and produces a flat partition; hierarchical produces a dendrogram showing all granularities and doesn't require K. K-means scales to very large datasets (O(nKp) per iteration); hierarchical is slow for large n (O(n² log n) or O(n³)). K-means is non-deterministic (random initialization); hierarchical is deterministic. Use K-means when: n is large, K is known or easily estimated, clusters are roughly spherical. Use hierarchical when: you want to explore multiple levels of granularity, n is moderate, or you want to understand the nested cluster structure.
