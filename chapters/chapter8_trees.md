# Chapter 8: Tree-Based Methods

Tree-based methods partition the feature space into rectangular regions and fit a simple model (usually a constant) in each region. They are highly flexible, handle mixed data types, require minimal preprocessing, and are the foundation of the most powerful ensemble methods.

---

## Key Concepts

### Decision Trees

A decision tree recursively splits the data by choosing the predictor and split point that best separates the response.

**For regression trees**: minimize RSS in each region. At each split, choose predictor $X_j$ and cutpoint $s$ to minimize:

$$\sum_{i: x_i \in R_1} (y_i - \hat{y}_{R_1})^2 + \sum_{i: x_i \in R_2} (y_i - \hat{y}_{R_2})^2$$

**For classification trees**: minimize **Gini impurity** or **entropy** (not misclassification rate, which is too coarse for growing):

$$G = \sum_{k=1}^K \hat{p}_{mk}(1 - \hat{p}_{mk}) \qquad \text{(Gini)}$$

$$H = -\sum_{k=1}^K \hat{p}_{mk} \log(\hat{p}_{mk}) \qquad \text{(Entropy/Deviance)}$$

where $\hat{p}_{mk}$ = proportion of class k observations in region m. Both measures are small when a node is pure (all one class).

**Prediction**: for regression, predict the mean of training observations in the leaf. For classification, predict the majority class.

---

### Tree Pruning (Cost-Complexity Pruning)

A fully grown tree overfits. **Cost-complexity pruning** controls tree size by penalizing complexity:

$$\sum_{m=1}^{|T|} \sum_{i: x_i \in R_m} (y_i - \hat{y}_{R_m})^2 + \alpha |T|$$

where $|T|$ = number of terminal nodes and $\alpha \geq 0$ is the tuning parameter.

- $\alpha = 0$: full unpruned tree
- Large $\alpha$: heavily penalizes complexity → small tree

Choose $\alpha$ by cross-validation. The result is a nested sequence of subtrees — pick the one with lowest CV error.

**Bias-variance tradeoff in trees**:
- Deep tree: low bias, high variance (overfits)
- Shallow tree: high bias, low variance (underfits)

---

### Bagging (Bootstrap Aggregation)

Trees have high variance — small changes in data can produce very different trees. **Bagging** reduces variance by averaging many trees:

1. Draw B bootstrap samples from the training data
2. Fit a full (unpruned) decision tree to each bootstrap sample
3. Average predictions (regression) or take majority vote (classification)

$$\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}^{*b}(x)$$

Bagging dramatically reduces variance with little increase in bias.

**Out-of-Bag (OOB) Error**: each bootstrap sample uses ~63% of the data. The remaining ~37% (out-of-bag observations) can be used to estimate test error **without a separate validation set**. OOB error is a valid estimate of the generalization error.

---

### Random Forests

The problem with bagging: if there is one very strong predictor, most trees will use it at the first split, making the trees highly correlated. Averaging correlated trees doesn't reduce variance as much.

**Random forests** decorrelate the trees by restricting each split to a **random subset of m predictors**:

- Default: $m = \sqrt{p}$ for classification, $m = p/3$ for regression
- When $m = p$: bagging (no restriction)
- Smaller m: more decorrelation, lower variance, potentially higher bias

The key insight: by excluding the dominant predictor from some splits, other predictors get a chance to structure the tree differently, producing diverse trees whose average is more stable.

**Variable Importance**: two measures
1. **Mean decrease in impurity**: total reduction in Gini/RSS from splits on variable j, averaged over all trees
2. **Mean decrease in accuracy (permutation importance)**: permute variable j's values in OOB data; measure increase in OOB error. More reliable.

---

### Boosting

Instead of building trees independently (bagging/RF), **boosting** builds trees sequentially — each tree fits the **residuals** from the previous trees.

**Algorithm (gradient boosting for regression)**:

1. Initialize: $\hat{f}(x) = 0$, $r_i = y_i$ for all i
2. For b = 1, 2, ..., B:
   a. Fit a tree $\hat{f}^b$ with d splits to the residuals $\{(x_i, r_i)\}$
   b. Update: $\hat{f}(x) \leftarrow \hat{f}(x) + \lambda \hat{f}^b(x)$
   c. Update residuals: $r_i \leftarrow r_i - \lambda \hat{f}^b(x_i)$
3. Final model: $\hat{f}(x) = \sum_{b=1}^B \lambda \hat{f}^b(x)$

**Key tuning parameters**:
- **B** (number of trees): larger B can overfit; choose by CV
- **λ** (shrinkage/learning rate): typically 0.01–0.1; smaller λ → more trees needed
- **d** (interaction depth): number of splits per tree; d=1 = stumps (additive model); d=2 = two-way interactions

**Boosting vs. Random Forests**:
- Boosting learns slowly (shrinkage), sequentially, fitting residuals → lower bias
- Random forests are fully grown trees, averaged → lower variance
- Boosting can overfit with large B; RF is more robust to B choice
- In practice, XGBoost/LightGBM (optimized gradient boosting) often outperforms RF

---

### BART (Bayesian Additive Regression Trees)

BART is a Bayesian ensemble: models Y as a sum of many small trees, each fitted to residuals, with priors that regularize each tree to be weak. Produces full posterior distributions over predictions — useful when uncertainty quantification matters.

---

### Summary Comparison

| Method | Variance | Bias | Interpretability | Overfitting Risk |
|---|---|---|---|---|
| Single tree | High | Low (deep) / High (shallow) | High (visual) | High |
| Bagging | Lower | Same as single tree | Low | Moderate |
| Random Forest | Low | Slightly higher | Low (importance plots) | Low |
| Boosting | Lowest | Low | Low | High (large B, small λ) |

---

## Real Data Example: Predicting Baseball Salary (Hitters) + Boston Housing

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               BaggingRegressor)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ── Load Hitters dataset (baseball salary prediction) ─────────────────────────
try:
    from ISLP import load_data
    Hitters = load_data("Hitters")
except ImportError:
    url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Hitters.csv"
    Hitters = pd.read_csv(url, index_col=0)

Hitters = Hitters.dropna()
Hitters["LogSalary"] = np.log(Hitters["Salary"])

# Encode categorical features
Hitters = pd.get_dummies(Hitters, columns=["League", "Division", "NewLeague"],
                          drop_first=True)

feature_cols = [c for c in Hitters.columns if c not in ["Salary", "LogSalary"]]
X = Hitters[feature_cols].astype(float)
y = Hitters["LogSalary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                      random_state=42)

# ── Single decision tree ──────────────────────────────────────────────────────
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(18, 6))
plot_tree(tree, feature_names=feature_cols, filled=True, ax=ax, fontsize=8)
ax.set_title("Decision Tree (max_depth=3): Baseball Salary")
plt.tight_layout()
plt.show()

tree_mse = mean_squared_error(y_test, tree.predict(X_test))
print(f"Single tree test MSE: {tree_mse:.4f}")

# ── Pruning via cross-validation ─────────────────────────────────────────────
depths = range(1, 15)
cv_scores = []
for d in depths:
    t = DecisionTreeRegressor(max_depth=d, random_state=42)
    cv = cross_val_score(t, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    cv_scores.append(-cv.mean())

best_depth = depths[np.argmin(cv_scores)]
print(f"Best depth by CV: {best_depth}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(depths), cv_scores, "o-", color="steelblue")
ax.axvline(best_depth, color="red", linestyle="--", label=f"Best depth = {best_depth}")
ax.set_xlabel("Tree depth")
ax.set_ylabel("CV MSE")
ax.set_title("Cross-Validation for Tree Depth")
ax.legend()
plt.tight_layout()
plt.show()

# ── Bagging ───────────────────────────────────────────────────────────────────
bag = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=500,
    oob_score=True,
    random_state=42
)
bag.fit(X_train, y_train)
bag_mse = mean_squared_error(y_test, bag.predict(X_test))
print(f"Bagging test MSE:     {bag_mse:.4f}")
print(f"Bagging OOB MSE:      {1 - bag.oob_score_:.4f}")

# ── Random Forest ─────────────────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
rf_mse = mean_squared_error(y_test, rf.predict(X_test))
print(f"Random Forest test MSE: {rf_mse:.4f}")
print(f"Random Forest OOB MSE:  {1 - rf.oob_score_:.4f}")

# Variable importance
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
importances[:10].plot(kind="bar", ax=ax, color="steelblue")
ax.set_ylabel("Mean Decrease in Impurity")
ax.set_title("Random Forest: Top 10 Variable Importances")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# Effect of mtry (number of features per split)
print("\nEffect of mtry on test MSE:")
p = X_train.shape[1]
for max_feat in [1, int(np.sqrt(p)), int(p/3), p]:
    rf_m = RandomForestRegressor(n_estimators=200, max_features=max_feat, random_state=42)
    rf_m.fit(X_train, y_train)
    mse_m = mean_squared_error(y_test, rf_m.predict(X_test))
    label = f"m={max_feat}"
    if max_feat == p: label += " (bagging)"
    if max_feat == int(np.sqrt(p)): label += " (sqrt, default)"
    print(f"  {label:30s}: MSE = {mse_m:.4f}")

# ── Gradient Boosting ─────────────────────────────────────────────────────────
# Effect of number of trees and learning rate
gb = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
gb_mse = mean_squared_error(y_test, gb.predict(X_test))
print(f"\nGradient Boosting test MSE: {gb_mse:.4f}")

# Plot test MSE vs. number of trees
test_mse_by_n = [
    mean_squared_error(y_test, pred)
    for pred in gb.staged_predict(X_test)
]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, len(test_mse_by_n)+1), test_mse_by_n, color="steelblue", linewidth=1)
ax.set_xlabel("Number of trees")
ax.set_ylabel("Test MSE")
ax.set_title("Gradient Boosting: Test MSE vs. Number of Trees\n(learning_rate=0.01)")
ax.axvline(np.argmin(test_mse_by_n)+1, color="red", linestyle="--",
           label=f"Optimal B = {np.argmin(test_mse_by_n)+1}")
ax.legend()
plt.tight_layout()
plt.show()

# ── Final comparison ──────────────────────────────────────────────────────────
print("\nModel comparison (test MSE on log salary):")
print(f"  Single tree (depth=3):  {tree_mse:.4f}")
print(f"  Bagging (B=500):        {bag_mse:.4f}")
print(f"  Random Forest (B=500):  {rf_mse:.4f}")
print(f"  Gradient Boosting:      {gb_mse:.4f}")
```

---

## Interview Questions

**Q1: What is the difference between bagging and random forests?**

Both build many decision trees on bootstrap samples and average the predictions. The difference: random forests restrict each split to a random subset of $m$ predictors (default $m = \sqrt{p}$), while bagging uses all $p$ predictors at every split. This restriction decorrelates the trees — when there's a dominant predictor, bagging trees are all similar (correlated), so averaging doesn't reduce variance much. By randomly excluding the dominant predictor from some splits, random forests produce more diverse trees that average to a lower-variance model. Bagging is a special case of random forests with $m = p$.

---

**Q2: What is out-of-bag (OOB) error and why is it useful?**

Each bootstrap sample uses roughly 63% of the training data. The remaining ~37% (out-of-bag observations) were not used to fit that particular tree. You can predict each training observation using only the trees for which it was OOB, then compute the error on these predictions. OOB error is a valid estimate of test error without needing a separate validation set. It's essentially a free cross-validation built into the bagging/RF procedure, which is why RF is often less sensitive to train/test splits than other methods.

---

**Q3: Why does random forest use $m = \sqrt{p}$ for classification and $p/3$ for regression?**

These are empirically validated defaults, not theoretically derived. The key insight is that m controls the bias-variance tradeoff of the ensemble: smaller m → more decorrelated trees → lower variance, but each tree is fit on less information → higher bias. $\sqrt{p}$ and $p/3$ have been found to work well across many datasets as a starting point. Always tune m by cross-validation — the optimal value is dataset-dependent, especially when a few predictors dominate.

---

**Q4: When would you prefer boosting over random forests?**

Boosting tends to achieve lower bias because it explicitly fits residuals — each tree corrects the mistakes of the previous ensemble. Random forests reduce variance by averaging, but bias is set by a single tree's depth. Prefer boosting when: (1) you need the best possible predictive accuracy and are willing to tune more hyperparameters; (2) the signal in the data is strong but complex; (3) you have enough data to support deep trees with small learning rates. Prefer random forests when: (1) you want a robust model with minimal tuning; (2) training time is constrained; (3) interpretability via variable importance matters; (4) overfitting is a concern (boosting can overfit badly with too many trees or too large a learning rate).

---

**Q5: What is variable importance in random forests, and what are its limitations?**

Two measures: (1) **Mean decrease in impurity (MDI)**: total reduction in Gini/RSS attributed to splits on variable j, averaged across all trees. Fast but biased toward high-cardinality variables. (2) **Permutation importance**: randomly permute variable j's values in the OOB data; measure the increase in OOB error. More reliable and less biased, but slower. Limitation: both measures give high importance to correlated variables — the importance is split between them, making each look less important than it really is. Use with caution when predictors are highly correlated.

---

**Q6: What are the advantages and disadvantages of decision trees compared to linear models?**

**Advantages**: (1) Naturally handle interactions and non-linearities without manual specification; (2) handle mixed data types (numeric and categorical) without preprocessing; (3) no need to scale features; (4) very interpretable (can visualize small trees); (5) robust to outliers (splits are based on rank order).

**Disadvantages**: (1) High variance — small changes in data produce very different trees; (2) do not extrapolate beyond the range of training data; (3) axis-aligned splits can be inefficient for linear relationships; (4) single trees are weak predictors. In practice, single trees are rarely used — always use ensembles (RF, boosting).

---

**Q7: What is the effect of the learning rate (shrinkage) in gradient boosting?**

The learning rate $\lambda$ scales each tree's contribution to the ensemble: smaller $\lambda$ → each tree makes a smaller update → more trees B needed to achieve the same training fit → better generalization because more regularization. There's a tradeoff: $\lambda = 0.01$ with $B = 5000$ trees typically outperforms $\lambda = 0.1$ with $B = 500$ trees, but takes 10x longer to train. In practice, use a small learning rate (0.01–0.05) with early stopping (stop when validation error stops decreasing) rather than trying to choose B manually.
