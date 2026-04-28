# Chapter 6: Linear Model Selection and Regularization

When p is large relative to n, or when many predictors are irrelevant, standard OLS performs poorly. Model selection and regularization methods address this by reducing variance at the cost of a small increase in bias.

---

## Key Concepts

### Subset Selection

**Best Subset Selection**: fit all $2^p$ models, choose by CV error (or AIC/BIC/Adjusted R²).
- Computationally infeasible for p > ~40
- High variance: searching many models inflates the chance of overfitting

**Forward Stepwise Selection**:
1. Start with null model (intercept only)
2. At each step, add the predictor that most improves fit
3. Choose final model size by CV/AIC/BIC
- Computationally efficient: fits O(p²) models
- May miss optimal models (greedy)

**Backward Stepwise Selection**:
1. Start with full model (all p predictors)
2. At each step, remove the least useful predictor
- Requires n > p (full model must be fit)

**Choosing the model size**:

| Criterion | Formula | Penalizes |
|---|---|---|
| Adjusted R² | $1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)}$ | Adding useless predictors |
| AIC | $\frac{1}{n\hat{\sigma}^2}(\text{RSS} + 2p\hat{\sigma}^2)$ | Number of parameters (2p) |
| BIC | $\frac{1}{n}(\text{RSS} + \log(n) \cdot p\hat{\sigma}^2)$ | log(n) per parameter — heavier penalty |
| CV error | Direct estimate of test error | N/A |

BIC penalizes complexity more than AIC (log(n) > 2 for n > 7), so tends to select smaller models. CV is the gold standard — it directly estimates test error.

---

### Ridge Regression

Adds an **L2 penalty** to the RSS:

$$\hat{\beta}^{\text{ridge}} = \arg\min_\beta \left\{ \text{RSS} + \lambda \sum_{j=1}^p \beta_j^2 \right\}$$

- **λ = 0**: OLS solution
- **λ → ∞**: all coefficients shrink to 0

**Properties**:
- All p predictors remain in the model (no variable selection)
- Shrinks coefficients toward zero, especially those of correlated predictors
- Closed-form solution: $\hat{\beta}^{\text{ridge}} = (X^TX + \lambda I)^{-1} X^T y$
- **Must standardize predictors** before applying ridge (coefficients are scale-dependent)
- Effective degrees of freedom: $\sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}$ where $d_j$ are singular values of X

**Why ridge works**: when predictors are correlated, OLS inflates coefficient variances. Ridge stabilizes them. In the bias-variance tradeoff: ridge accepts a little bias to get a large variance reduction.

---

### The Lasso

Adds an **L1 penalty** to the RSS:

$$\hat{\beta}^{\text{lasso}} = \arg\min_\beta \left\{ \text{RSS} + \lambda \sum_{j=1}^p |\beta_j| \right\}$$

**Key difference from ridge**: Lasso produces **sparse solutions** — some coefficients are exactly zero. Lasso performs **variable selection**.

**Why L1 gives sparsity**: the L1 constraint region has corners at the axes. The RSS contours hit these corners first, setting coefficients exactly to zero. L2 (ridge) has a round constraint region — no corners, so coefficients shrink but rarely hit exactly zero.

**Bias-variance tradeoff**:
- Small λ: low bias, high variance (close to OLS)
- Large λ: high bias, low variance (coefficients shrunk to zero)
- Optimal λ chosen by cross-validation

**When Lasso > Ridge**: when the true model is sparse (few predictors matter). Lasso picks them out.
**When Ridge > Lasso**: when many predictors contribute small, real effects. Ridge shrinks all evenly rather than zeroing most.

---

### Elastic Net

Combines L1 and L2 penalties:

$$\hat{\beta}^{\text{elastic}} = \arg\min_\beta \left\{ \text{RSS} + \lambda_1 \sum_j |\beta_j| + \lambda_2 \sum_j \beta_j^2 \right\}$$

Or equivalently (sklearn parameterization):

$$\text{RSS} + \lambda \left[\alpha \sum_j |\beta_j| + \frac{1-\alpha}{2} \sum_j \beta_j^2\right]$$

where $\alpha \in [0,1]$ controls the L1/L2 mix. α=1 = Lasso, α=0 = Ridge.

**When to use**: high-dimensional data with correlated predictors — Lasso tends to randomly pick one from a correlated group; elastic net keeps them all (Ridge's behavior) while still doing selection (Lasso's behavior).

---

### Principal Components Regression (PCR)

1. Compute the first M principal components $Z_1, \ldots, Z_M$ from the predictors X
2. Regress Y on $Z_1, \ldots, Z_M$

**Key idea**: PC's capture the directions of maximum variance in X. If the assumption holds that the directions with most variance in X are also the most informative for Y, PCR works well.

**Caveat**: PCR is **unsupervised** — the components are chosen to explain variance in X, not Y. If the low-variance directions are highly predictive of Y, PCR will miss them.

- M = p: equivalent to OLS
- M = 1: most compressed, highest bias
- Choose M by cross-validation
- Must standardize predictors before PCA

---

### Partial Least Squares (PLS)

Like PCR but **supervised**: constructs directions that explain variance in Y as well as X.

1. First direction: each predictor weighted by its simple regression coefficient with Y
2. Subsequent directions: orthogonalize and repeat

In practice, PCR and PLS often give similar results. PLS may perform better when a small number of predictors are strongly correlated with Y.

---

### Why Regularization Reduces Test Error

OLS minimizes training RSS — it can overfit when p is large. Regularization introduces bias but reduces variance:

$$\text{Test Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$$

Regularization reduces variance (shrinking coefficients reduces their sensitivity to the training data) at the cost of a small increase in bias. At the optimal λ, the variance reduction outweighs the bias increase.

---

## Real Data Example: Hitters Dataset (Salary Prediction)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    from ISLP import load_data
    Hitters = load_data("Hitters")
except ImportError:
    url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Hitters.csv"
    Hitters = pd.read_csv(url, index_col=0)

Hitters = Hitters.dropna()
Hitters["LogSalary"] = np.log(Hitters["Salary"])
Hitters = pd.get_dummies(Hitters, columns=["League", "Division", "NewLeague"], drop_first=True)

feature_cols = [c for c in Hitters.columns if c not in ["Salary", "LogSalary"]]
X = Hitters[feature_cols].astype(float).values
y = Hitters["LogSalary"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Ridge: coefficient paths ──────────────────────────────────────────────────
alphas = np.logspace(-3, 6, 200)
ridge_coefs = []
for a in alphas:
    r = Ridge(alpha=a)
    r.fit(X_train_s, y_train)
    ridge_coefs.append(r.coef_)

ridge_coefs = np.array(ridge_coefs)

fig, ax = plt.subplots(figsize=(10, 5))
for j in range(ridge_coefs.shape[1]):
    ax.plot(np.log10(alphas), ridge_coefs[:, j], alpha=0.5)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("log10(λ)")
ax.set_ylabel("Coefficient value")
ax.set_title("Ridge: Coefficient Paths")
plt.tight_layout()
plt.show()

# ── Ridge CV ──────────────────────────────────────────────────────────────────
kf = KFold(n_splits=10, shuffle=True, random_state=42)
ridge_cv_mse = []
for a in alphas:
    scores = cross_val_score(Ridge(alpha=a), X_train_s, y_train,
                              cv=kf, scoring="neg_mean_squared_error")
    ridge_cv_mse.append(-scores.mean())

best_ridge_alpha = alphas[np.argmin(ridge_cv_mse)]
print(f"Best Ridge λ (CV): {best_ridge_alpha:.4f}")

ridge_final = Ridge(alpha=best_ridge_alpha)
ridge_final.fit(X_train_s, y_train)
ridge_test_mse = mean_squared_error(y_test, ridge_final.predict(X_test_s))
print(f"Ridge test MSE: {ridge_test_mse:.4f}")

# ── Lasso: coefficient paths + CV ────────────────────────────────────────────
lasso_alphas = np.logspace(-4, 1, 200)
lasso_coefs = []
for a in lasso_alphas:
    l = Lasso(alpha=a, max_iter=10000)
    l.fit(X_train_s, y_train)
    lasso_coefs.append(l.coef_)

lasso_coefs = np.array(lasso_coefs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for j in range(lasso_coefs.shape[1]):
    axes[0].plot(np.log10(lasso_alphas), lasso_coefs[:, j], alpha=0.5)
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_xlabel("log10(λ)")
axes[0].set_ylabel("Coefficient value")
axes[0].set_title("Lasso: Coefficient Paths (many → 0)")

lasso_cv_mse = []
for a in lasso_alphas:
    scores = cross_val_score(Lasso(alpha=a, max_iter=10000), X_train_s, y_train,
                              cv=kf, scoring="neg_mean_squared_error")
    lasso_cv_mse.append(-scores.mean())

axes[1].semilogx(lasso_alphas, lasso_cv_mse, "o-", color="steelblue", markersize=3)
best_lasso_alpha = lasso_alphas[np.argmin(lasso_cv_mse)]
axes[1].axvline(best_lasso_alpha, color="red", linestyle="--",
                label=f"Best λ = {best_lasso_alpha:.4f}")
axes[1].set_xlabel("λ (log scale)")
axes[1].set_ylabel("10-Fold CV MSE")
axes[1].set_title("Lasso: CV for λ Selection")
axes[1].legend()
plt.tight_layout()
plt.show()

lasso_final = Lasso(alpha=best_lasso_alpha, max_iter=10000)
lasso_final.fit(X_train_s, y_train)
lasso_test_mse = mean_squared_error(y_test, lasso_final.predict(X_test_s))
n_nonzero = np.sum(lasso_final.coef_ != 0)
print(f"\nBest Lasso λ (CV): {best_lasso_alpha:.4f}")
print(f"Lasso non-zero coefficients: {n_nonzero} / {X.shape[1]}")
print(f"Lasso test MSE: {lasso_test_mse:.4f}")

# ── PCR ───────────────────────────────────────────────────────────────────────
pcr_cv_mse = []
for m in range(1, X_train_s.shape[1] + 1):
    pipe = Pipeline([("pca", PCA(n_components=m)), ("lr", LinearRegression())])
    scores = cross_val_score(pipe, X_train_s, y_train,
                             cv=kf, scoring="neg_mean_squared_error")
    pcr_cv_mse.append(-scores.mean())

best_m = np.argmin(pcr_cv_mse) + 1
print(f"\nPCR: best M = {best_m} components")

pcr_final = Pipeline([("pca", PCA(n_components=best_m)), ("lr", LinearRegression())])
pcr_final.fit(X_train_s, y_train)
pcr_test_mse = mean_squared_error(y_test, pcr_final.predict(X_test_s))
print(f"PCR test MSE: {pcr_test_mse:.4f}")

# ── Summary comparison ────────────────────────────────────────────────────────
ols = LinearRegression().fit(X_train_s, y_train)
ols_mse = mean_squared_error(y_test, ols.predict(X_test_s))

print(f"\nModel comparison (test MSE on log salary):")
print(f"  OLS:   {ols_mse:.4f}")
print(f"  Ridge: {ridge_test_mse:.4f}")
print(f"  Lasso: {lasso_test_mse:.4f} ({n_nonzero}/{X.shape[1]} predictors)")
print(f"  PCR:   {pcr_test_mse:.4f} ({best_m} components)")
```

---

## Interview Questions

**Q1: What is the bias-variance tradeoff in ridge regression?**

OLS minimizes training RSS — it has zero bias but high variance when predictors are correlated or p is close to n. Ridge adds a penalty $\lambda \sum \beta_j^2$ that shrinks coefficients toward zero, introducing a small bias. But the shrinkage reduces variance substantially — the coefficients are less sensitive to noise in the training data. At the optimal λ (chosen by CV), the reduction in variance outweighs the increase in bias, giving lower test error than OLS. As λ → 0, ridge → OLS; as λ → ∞, all coefficients → 0.

---

**Q2: What is the difference between Ridge and Lasso, and when would you prefer each?**

Both add a penalty to RSS, but Ridge uses L2 ($\sum \beta_j^2$) and Lasso uses L1 ($\sum |\beta_j|$). The key practical difference: Lasso produces sparse solutions (many coefficients exactly zero) — it performs variable selection. Ridge shrinks all coefficients but rarely zeros them out.

Use Lasso when: the true model is sparse (few predictors truly matter) and you want automatic variable selection. Use Ridge when: many predictors contribute small real effects; or predictors are highly correlated (Lasso arbitrarily picks one from a correlated group, Ridge keeps them all shrunk). Elastic net is a hybrid when you want selection but with correlated predictors.

---

**Q3: Why does Lasso produce sparse solutions but Ridge doesn't?**

The geometry: ridge constrains coefficients to a sphere ($\sum \beta_j^2 \leq t$) and lasso to a diamond ($\sum |\beta_j| \leq t$). The RSS contours (ellipses) intersect the constraint region at the first point they touch. The diamond has corners at the coordinate axes — the RSS contours almost always hit these corners first, setting one or more coefficients to exactly zero. The sphere has no corners, so the intersection rarely falls exactly on an axis.

---

**Q4: How do you choose the regularization parameter λ?**

Use cross-validation. For a grid of λ values, compute k-fold CV error. Choose the λ that minimizes CV error, or (using the 1-SE rule) the largest λ whose CV error is within 1 SE of the minimum — favoring more regularization when the difference is within noise. In sklearn, `RidgeCV` and `LassoCV` do this efficiently. For Lasso, the solution path is piecewise linear (LARS algorithm), so the entire path can be computed efficiently.

---

**Q5: What is Principal Components Regression, and when does it fail?**

PCR projects predictors onto their first M principal components (directions of maximum variance in X), then regresses Y on those components. It works well when the assumption holds: the directions of high variance in X are also informative for Y. It fails when: important predictors have low variance in the training data (PCR would discard them); or when the relevant directions for predicting Y are not the same as the high-variance directions of X. Unlike Ridge/Lasso, PCR discards the low-variance components entirely rather than shrinking them.

---

**Q6: What are the advantages of forward stepwise over best subset selection?**

Best subset selection considers all $2^p$ possible models — this is computationally infeasible for p > ~40 and has very high variance (searching among many models inflates the chance of finding a good fit by chance). Forward stepwise considers only O(p²) models, is computationally feasible for large p, and tends to have lower variance because the search space is smaller. The tradeoff: forward stepwise may miss the optimal model (it's greedy — adding a predictor that's individually best may prevent finding the globally best combination).

---

**Q7: Why must you standardize predictors before applying Ridge or Lasso?**

Ridge and Lasso penalize the size of coefficients. If predictors are on different scales (e.g., income in dollars vs. age in years), a coefficient's magnitude depends on the scale of the predictor — penalizing large coefficients equally would unfairly penalize predictors measured in small units. Standardizing to zero mean and unit variance puts all predictors on equal footing so the penalty treats them fairly. This is not needed for OLS because the penalty on coefficient magnitude doesn't affect OLS.
