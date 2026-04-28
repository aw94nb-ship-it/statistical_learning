# Chapter 5: Resampling Methods

Resampling methods repeatedly draw samples from a training set and refit a model on each sample. They estimate test error and quantify uncertainty — two tasks that are otherwise difficult without a large held-out test set.

---

## Key Concepts

### The Validation Set Approach

Split data randomly into a **training set** and a **validation (hold-out) set**. Fit the model on training, evaluate on validation.

**Limitations**:
- High variance: the test error estimate depends heavily on which observations land in each split
- Wasteful: uses only half the data for training → tends to overestimate test error for models fit on the full dataset

---

### Leave-One-Out Cross-Validation (LOOCV)

For each observation $i = 1, \ldots, n$:
1. Fit the model on all observations **except** $i$
2. Predict $\hat{y}_i$ for observation $i$
3. Compute squared error $(y_i - \hat{y}_i)^2$

$$\text{CV}_{(n)} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Advantages**: low bias (trains on nearly the full dataset each time), deterministic (no randomness).

**Disadvantages**: computationally expensive for large n; high variance because the n training sets are nearly identical (highly correlated estimates).

**Shortcut for linear models**: LOOCV can be computed in one fit using the leverage statistic $h_i$:

$$\text{CV}_{(n)} = \frac{1}{n} \sum_{i=1}^n \left(\frac{y_i - \hat{y}_i}{1 - h_i}\right)^2$$

---

### k-Fold Cross-Validation

Divide data into k roughly equal folds. For each fold $j$:
1. Fit model on all folds except $j$
2. Predict on fold $j$
3. Compute error on fold $j$

$$\text{CV}_{(k)} = \frac{1}{k} \sum_{j=1}^k \text{MSE}_j$$

Typical choices: **k = 5** or **k = 10**.

**Bias-variance tradeoff**:
- LOOCV (k = n): low bias, high variance, expensive
- k = 5 or 10: intermediate bias, lower variance, fast
- k = 2: high bias (trains on 50%), low variance

**Why k-fold often outperforms LOOCV in practice**: LOOCV has higher variance because each training set is almost identical, making the n error estimates highly correlated. Averaging highly correlated quantities reduces variance less than averaging more independent quantities (k-fold folds are more different from each other).

---

### Classification Setting

For classification, use the **misclassification rate** instead of MSE:

$$\text{CV}_{(k)} = \frac{1}{k} \sum_{j=1}^k \frac{1}{n_j} \sum_{i \in \text{fold}_j} I(y_i \neq \hat{y}_i)$$

---

### The 1-Standard-Error Rule

When using CV to choose a tuning parameter, don't just pick the minimum CV error. The **1-SE rule**: choose the **simplest model** whose CV error is within 1 standard error of the minimum. This favors parsimony without sacrificing much accuracy.

---

### The Bootstrap

The bootstrap estimates the **variability** of a statistic (or model parameter) by repeatedly resampling the training data **with replacement**.

1. Draw B bootstrap samples of size n from the original data (with replacement)
2. Compute the statistic of interest on each sample: $\hat{\alpha}^{*1}, \ldots, \hat{\alpha}^{*B}$
3. Estimate SE:

$$\widehat{\text{SE}}(\hat{\alpha}) = \sqrt{\frac{1}{B-1} \sum_{b=1}^B \left(\hat{\alpha}^{*b} - \frac{1}{B}\sum_{b'=1}^B \hat{\alpha}^{*b'}\right)^2}$$

Each bootstrap sample uses ~63% of the original observations (some appear multiple times, ~37% don't appear at all — the OOB observations).

**What bootstrap is good for**:
- Estimating SE of any statistic, even when the theoretical SE is hard to derive
- Confidence intervals via bootstrap percentile method
- OOB error in bagging/random forests (free cross-validation)

**Bootstrap is NOT the same as cross-validation**: CV estimates test error (how well will the model predict new data?). Bootstrap estimates variability of an estimator (how much does this statistic fluctuate across different datasets?).

**Bootstrap for test error estimation has a problem**: bootstrap samples overlap significantly with the training data (~63% of observations appear in each). A model fit on bootstrap data predicts on some of its training observations, underestimating test error. This is why CV (not bootstrap) is preferred for test error estimation.

---

### CV for Model Selection vs. Model Assessment

Two distinct uses:
- **Model selection**: use CV to choose the best model/tuning parameter (e.g., best polynomial degree, best λ for ridge)
- **Model assessment**: estimate the final model's test error

When both are needed, use **nested CV**: outer loop for assessment, inner loop for selection. Alternatively, split data into three sets: train / validation (for selection) / test (for final assessment). Never use the test set during model selection.

---

## Real Data Example: Auto Dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import (cross_val_score, KFold, LeaveOneOut,
                                      train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

# ── Load Auto dataset ─────────────────────────────────────────────────────────
try:
    from ISLP import load_data
    Auto = load_data("Auto")
except ImportError:
    url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Auto.csv"
    Auto = pd.read_csv(url, na_values="?").dropna()

X = Auto[["horsepower"]].values
y = Auto["mpg"].values

# ── Validation set approach ───────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

val_mse = []
for d in range(1, 6):
    pipe = Pipeline([("poly", PolynomialFeatures(d)), ("lr", LinearRegression())])
    pipe.fit(X_train, y_train)
    mse = mean_squared_error(y_val, pipe.predict(X_val))
    val_mse.append(mse)
    print(f"Degree {d}: validation MSE = {mse:.2f}")

# ── LOOCV ─────────────────────────────────────────────────────────────────────
loo_mse = []
loo = LeaveOneOut()
for d in range(1, 6):
    pipe = Pipeline([("poly", PolynomialFeatures(d)), ("lr", LinearRegression())])
    scores = cross_val_score(pipe, X, y, cv=loo, scoring="neg_mean_squared_error")
    loo_mse.append(-scores.mean())

# ── 10-Fold CV ────────────────────────────────────────────────────────────────
kfold_mse = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for d in range(1, 6):
    pipe = Pipeline([("poly", PolynomialFeatures(d)), ("lr", LinearRegression())])
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error")
    kfold_mse.append(-scores.mean())

# ── Plot all three ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
degrees = range(1, 6)
ax.plot(degrees, val_mse,   "o-", label="Validation set (single split)", color="gray")
ax.plot(degrees, loo_mse,   "s-", label="LOOCV",                        color="coral")
ax.plot(degrees, kfold_mse, "^-", label="10-Fold CV",                   color="steelblue")
ax.set_xlabel("Polynomial Degree")
ax.set_ylabel("MSE")
ax.set_title("Cross-Validation: Estimating Test Error for Polynomial Regression")
ax.legend()
plt.tight_layout()
plt.show()

print(f"\nOptimal degree (LOOCV):   {np.argmin(loo_mse) + 1}")
print(f"Optimal degree (10-fold): {np.argmin(kfold_mse) + 1}")

# ── 1-Standard-Error Rule ─────────────────────────────────────────────────────
kfold_mse_arr = []
kfold_se_arr = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for d in range(1, 11):
    pipe = Pipeline([("poly", PolynomialFeatures(d)), ("lr", LinearRegression())])
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error")
    kfold_mse_arr.append(-scores.mean())
    kfold_se_arr.append(scores.std() / np.sqrt(10))

min_idx = np.argmin(kfold_mse_arr)
threshold = kfold_mse_arr[min_idx] + kfold_se_arr[min_idx]
one_se_degree = next(d for d, mse in enumerate(kfold_mse_arr, 1) if mse <= threshold)
print(f"\n1-SE Rule: best degree = {min_idx+1}, 1-SE parsimonious = {one_se_degree}")

# ── Bootstrap: estimate SE of OLS coefficient ─────────────────────────────────
B = 1000
boot_coefs = []
rng = np.random.default_rng(42)

for _ in range(B):
    idx = rng.integers(0, len(X), size=len(X))
    X_b, y_b = X[idx], y[idx]
    lr = LinearRegression().fit(X_b, y_b)
    boot_coefs.append(lr.coef_[0])

boot_coefs = np.array(boot_coefs)
print(f"\nBootstrap SE of horsepower coefficient: {boot_coefs.std():.4f}")
print(f"Bootstrap 95% CI: [{np.percentile(boot_coefs, 2.5):.4f}, {np.percentile(boot_coefs, 97.5):.4f}]")

# Compare to analytical SE
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
X_sm = sm.add_constant(X)
ols = OLS(y, X_sm).fit()
print(f"Analytical SE of horsepower coefficient: {ols.bse[1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(boot_coefs, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
ax.axvline(boot_coefs.mean(), color="coral", linewidth=2, label=f"Mean = {boot_coefs.mean():.4f}")
ax.axvline(np.percentile(boot_coefs, 2.5), color="gray", linestyle="--")
ax.axvline(np.percentile(boot_coefs, 97.5), color="gray", linestyle="--", label="95% CI")
ax.set_xlabel("Horsepower coefficient")
ax.set_ylabel("Frequency")
ax.set_title("Bootstrap Distribution of OLS Coefficient")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Interview Questions

**Q1: What is the difference between LOOCV and k-fold CV?**

LOOCV is a special case of k-fold CV where k = n — every observation gets its own fold and the model is trained on n−1 observations each time. LOOCV has low bias (nearly all data used for training) but high variance (n training sets are nearly identical → highly correlated error estimates → averaging them doesn't reduce variance much). k-fold CV with k = 5 or 10 has slightly higher bias but lower variance because the training sets differ more from each other. Computationally, LOOCV requires fitting the model n times vs. k times; for linear models, LOOCV has a cheap shortcut via leverage statistics.

---

**Q2: What is the bootstrap and what can it estimate?**

The bootstrap estimates the variability of a statistic by repeatedly resampling the data with replacement. Each bootstrap sample has the same size as the original data but contains duplicates (~63% unique observations, ~37% left out). Compute the statistic of interest on each sample, then use the distribution across B samples to estimate standard errors and confidence intervals. The bootstrap works for any statistic, even when theoretical SEs are hard to derive (e.g., median, ratio of two coefficients, AUC). It is not designed for estimating test error — CV is better for that because bootstrap samples overlap too heavily with training data.

---

**Q3: Why is the validation set approach problematic?**

Two main issues. First, **high variance**: test error depends heavily on which observations end up in the validation set — the estimate varies substantially across different random splits. Second, **pessimistic bias**: the model is trained on only half the data, but the goal is to estimate test error for a model trained on the full dataset. Larger training sets generally give better models, so validation-set error overestimates the true test error. CV addresses both: k-fold uses more of the data for training (reducing bias) and averages over k splits (reducing variance).

---

**Q4: What is the 1-standard-error rule and when would you use it?**

When using CV to select a tuning parameter, the 1-SE rule says: rather than choosing the parameter value with the single lowest CV error, choose the simplest model (most regularized, lowest complexity) whose CV error is within 1 standard error of the minimum. The rationale: CV estimates are noisy, so differences smaller than 1 SE may not be meaningful. By favoring simplicity within the noise band, you get a more parsimonious model that's likely to generalize just as well. Used frequently in regularization selection (Lasso, Ridge) and polynomial degree selection.

---

**Q5: How should you use CV for model selection AND model assessment?**

These are two different goals and require separate data. If you use the same CV fold to both select the model and estimate its error, you get an overly optimistic error estimate (the model was selected to minimize that very error). The correct approach: (1) **nested CV**: outer loop for error estimation, inner loop for model selection; (2) **three-way split**: train/validation/test — use validation for selection, test for final assessment. The test set must never be used during model selection. Breaking this rule is a form of data leakage and will overestimate real-world performance.

---

**Q6: What does it mean that each bootstrap sample uses only ~63% of the original data?**

When drawing n samples with replacement from n observations, the probability that any specific observation is not selected in a single draw is (1 − 1/n). Over n draws, the probability it is never selected is (1 − 1/n)^n → e^{−1} ≈ 0.368 as n → ∞. So on average, each bootstrap sample leaves out about 37% of the original observations. These out-of-bag (OOB) observations can be used to estimate test error without a separate validation set — this is how OOB error in random forests is computed.
