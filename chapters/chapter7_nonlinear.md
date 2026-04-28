# Chapter 7: Moving Beyond Linearity

Linear models are interpretable but limited. When the true relationship between X and Y is non-linear, we can extend linear regression using **basis functions** — transformations of X that allow the model to capture non-linear patterns while still fitting a linear model in the transformed features.

---

## Key Concepts

### Polynomial Regression

Replace $X$ with $X, X^2, \ldots, X^d$:

$$y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \cdots + \beta_d x_i^d + \varepsilon_i$$

Still a linear model in the coefficients — OLS applies directly.

**Prediction**:

$$\hat{f}(x) = \hat{\beta}_0 + \hat{\beta}_1 x + \hat{\beta}_2 x^2 + \cdots + \hat{\beta}_d x^d$$

**Limitations**:
- Global: the polynomial fits the entire range with a single function — wiggles in one region affect all others
- Boundary behavior is poor: polynomials tend to oscillate wildly near the edges of the data
- High degrees are numerically unstable and hard to interpret

**Choosing d**: use CV or AIC. Rarely useful beyond d = 3 or 4.

---

### Step Functions (Piecewise Constants)

Divide X into K bins using cutpoints $c_1 < c_2 < \cdots < c_K$. Create K indicator variables:

$$C_k(x) = I(c_{k-1} \leq x < c_k), \quad k = 1, \ldots, K$$

Fit: $y_i = \beta_0 + \beta_1 C_1(x_i) + \cdots + \beta_K C_K(x_i) + \varepsilon_i$

Predictions are step functions — constant within each bin. Simple but very restrictive.

---

### Regression Splines

**Piecewise polynomials**: fit a separate polynomial in each region defined by **knots** $\xi_1, \ldots, \xi_K$.

**Constraint**: require the function to be **continuous** at each knot (and also have continuous first and second derivatives for cubic splines).

**Cubic spline**: piecewise cubic polynomials with continuous first and second derivatives at each knot. A cubic spline with K knots has $4 + K$ basis functions (degrees of freedom):

The **truncated power basis**: for each knot $\xi_k$:

$$h(x, \xi_k) = (x - \xi_k)_+^3 = \begin{cases} (x - \xi_k)^3 & \text{if } x > \xi_k \\ 0 & \text{otherwise} \end{cases}$$

Fit: $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \sum_{k=1}^K \theta_k h(x, \xi_k) + \varepsilon$

**Natural cubic splines**: add the constraint that the function is linear beyond the outermost knots. This reduces overfitting at the boundaries (where data is sparse) and reduces df from $4+K$ to $2+K$.

---

### Smoothing Splines

Rather than choosing knots, smoothing splines minimize:

$$\sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int [f''(t)]^2 \, dt$$

- The first term: RSS (fit to data)
- The second term: penalizes roughness (curvature)
- $\lambda$: controls smoothness; $\lambda = 0$ → interpolates data; $\lambda \to \infty$ → straight line

The solution is a **natural cubic spline** with a knot at every unique $x_i$. The effective degrees of freedom is a function of λ — choose by CV.

**Advantage over regression splines**: no need to choose knot locations — just choose λ.

---

### Local Regression (LOESS/LOWESS)

At each point $x_0$, fit a weighted polynomial using only the nearby observations:

1. Find the s = k/n fraction of training points nearest to $x_0$ (the "span")
2. Assign weights: observations closer to $x_0$ get higher weight (kernel function)
3. Fit weighted least squares polynomial at $x_0$

**Tuning parameter**: span s. Larger s → smoother (higher bias, lower variance). Choose by CV.

**Key property**: local regression can adapt to different local behaviors of f — the fit in one region doesn't constrain another. Good for data with varying local characteristics.

**Limitation**: doesn't generalize naturally to multiple predictors (curse of dimensionality makes "nearby" meaningless in high dimensions).

---

### Generalized Additive Models (GAMs)

Extend multiple linear regression by replacing each linear term with a flexible function:

$$y_i = \beta_0 + f_1(x_{i1}) + f_2(x_{i2}) + \cdots + f_p(x_{ip}) + \varepsilon_i$$

Each $f_j$ can be a spline, local regression, polynomial, or even just a linear term. The model is **additive** — each predictor contributes independently.

**Advantages**:
- Captures non-linear effects of each predictor automatically
- Maintains the additive structure → interpretable (plot $f_j$ vs. $X_j$ holding others fixed)
- Works for both regression and classification (logistic GAM)

**Limitation**: interactions between predictors require manual specification (just like linear models).

**Fitting**: backfitting algorithm — iteratively fit each $f_j$ on the residuals from all other terms.

**For classification**:

$$\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + f_1(x_1) + \cdots + f_p(x_p)$$

---

## Real Data Example: Wage Dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf

# ── Load Wage dataset ─────────────────────────────────────────────────────────
try:
    from ISLP import load_data
    Wage = load_data("Wage")
except ImportError:
    url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Wage.csv"
    Wage = pd.read_csv(url, index_col=0)

X_age = Wage[["age"]].values
y_wage = Wage["wage"].values

# ── Polynomial regression: degree selection by CV ─────────────────────────────
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_mse = []
for d in range(1, 11):
    pipe = Pipeline([("poly", PolynomialFeatures(d)),
                     ("lr", LinearRegression())])
    scores = cross_val_score(pipe, X_age, y_wage, cv=kf,
                             scoring="neg_mean_squared_error")
    cv_mse.append(-scores.mean())

best_d = np.argmin(cv_mse) + 1
print(f"Best polynomial degree by CV: {best_d}")

# ── Fit and plot polynomial regression ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

age_range = np.linspace(Wage["age"].min(), Wage["age"].max(), 300).reshape(-1, 1)

for d, color, label in [(1, "gray", "Linear"), (4, "coral", "Degree-4 poly")]:
    pipe = Pipeline([("poly", PolynomialFeatures(d)), ("lr", LinearRegression())])
    pipe.fit(X_age, y_wage)
    axes[0].plot(age_range, pipe.predict(age_range), linewidth=2,
                 color=color, label=label)

axes[0].scatter(Wage["age"], Wage["wage"], alpha=0.1, color="steelblue", s=10)
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Wage ($K)")
axes[0].set_title("Polynomial Regression: Wage ~ Age")
axes[0].legend()

axes[1].plot(range(1, 11), cv_mse, "o-", color="steelblue")
axes[1].axvline(best_d, color="red", linestyle="--", label=f"Best d={best_d}")
axes[1].set_xlabel("Polynomial Degree")
axes[1].set_ylabel("CV MSE")
axes[1].set_title("Degree Selection by 10-Fold CV")
axes[1].legend()
plt.tight_layout()
plt.show()

# ── Regression splines via statsmodels ───────────────────────────────────────
# Natural cubic spline using cr() in statsmodels formula
model_ns = smf.ols("wage ~ cr(age, df=4)", data=Wage).fit()
model_poly4 = smf.ols("wage ~ age + np.power(age,2) + np.power(age,3) + np.power(age,4)",
                       data=Wage).fit()

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(Wage["age"], Wage["wage"], alpha=0.1, color="steelblue", s=10)

age_pred = pd.DataFrame({"age": np.linspace(18, 80, 300)})
ax.plot(age_pred["age"], model_ns.predict(age_pred),
        color="coral", linewidth=2, label="Natural cubic spline (df=4)")
ax.plot(age_pred["age"], model_poly4.predict(age_pred),
        color="green", linewidth=2, label="Degree-4 polynomial")
ax.set_xlabel("Age")
ax.set_ylabel("Wage ($K)")
ax.set_title("Natural Cubic Spline vs Polynomial")
ax.legend()
plt.tight_layout()
plt.show()

# ── GAM with multiple predictors ─────────────────────────────────────────────
# Fit additive model: wage ~ s(age) + s(year) + education
# Using statsmodels with natural splines
model_gam = smf.ols(
    "wage ~ cr(age, df=4) + cr(year, df=4) + C(education)",
    data=Wage
).fit()

print(model_gam.summary())
print(f"\nGAM R²: {model_gam.rsquared:.4f}")

# ── Logistic GAM: P(wage > 250K) ─────────────────────────────────────────────
Wage["high_earn"] = (Wage["wage"] > 250).astype(int)
logit_gam = smf.logit("high_earn ~ cr(age, df=4) + cr(year, df=4) + C(education)",
                       data=Wage).fit(disp=0)

age_pred_df = pd.DataFrame({
    "age": np.linspace(18, 80, 300),
    "year": 2006,
    "education": "4. College Grad"
})

log_odds = logit_gam.predict(age_pred_df)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(age_pred_df["age"], log_odds, color="coral", linewidth=2)
ax.set_xlabel("Age")
ax.set_ylabel("P(Wage > $250K)")
ax.set_title("Logistic GAM: Probability of High Earner vs Age\n(year=2006, education=College Grad)")
plt.tight_layout()
plt.show()

# ── Step function ─────────────────────────────────────────────────────────────
Wage["age_cut"] = pd.cut(Wage["age"], bins=4)
model_step = smf.ols("wage ~ C(age_cut)", data=Wage).fit()
print(f"\nStep function R²: {model_step.rsquared:.4f}")
print(model_step.params)
```

---

## Interview Questions

**Q1: What is the difference between polynomial regression and regression splines?**

Polynomial regression fits a single polynomial of degree d globally across all X values — a degree-4 polynomial uses the same 4 coefficients everywhere. This causes problems at the extremes (oscillation) and means local changes require adjusting the global function. Regression splines fit separate polynomials in regions between **knots**, with smoothness constraints (continuous first and second derivatives for cubic splines). Splines are more flexible locally — the fit in one region doesn't distort another. Natural cubic splines add the constraint that the function is linear beyond the outermost knots, preventing boundary oscillation.

---

**Q2: What is a GAM and what are its limitations?**

A Generalized Additive Model replaces each linear term in a regression with a flexible function $f_j(X_j)$: $Y = \beta_0 + f_1(X_1) + f_2(X_2) + \ldots + f_p(X_p)$. Each $f_j$ can be a spline, polynomial, or local regression — allowing non-linear effects of each predictor. The model remains interpretable: you can plot each $f_j$ to see its contribution.

Limitation: GAMs are **additive** — they assume each predictor contributes independently to Y. Interactions between predictors must be added manually (e.g., $f_{12}(X_1, X_2)$). If the true model has important interactions not specified, GAMs will miss them. For complex interactions, tree-based methods (which capture interactions automatically) may be better.

---

**Q3: What is the tuning parameter in smoothing splines, and how do you choose it?**

The smoothing parameter λ controls the tradeoff between data fit and smoothness: $\min_f \sum (y_i - f(x_i))^2 + \lambda \int [f''(t)]^2 dt$. The second term penalizes curvature — large λ forces f toward a straight line; λ = 0 interpolates all data points. In practice, choose λ by cross-validation (minimizing CV MSE). Equivalently, the effective degrees of freedom (df) parameterizes λ: df = n corresponds to interpolation, df = 2 corresponds to a straight line. Practitioners often choose df by CV instead of λ directly.

---

**Q4: What is local regression (LOESS) and when is it useful?**

Local regression fits a weighted polynomial at each prediction point $x_0$ using only nearby training observations — closer observations get higher weights. The tuning parameter is the span: the fraction of observations used to fit each local model. Larger span → smoother fit (higher bias, lower variance). Local regression is useful when: the relationship between X and Y changes in different parts of the range; you don't want to impose any global functional form. Limitation: doesn't generalize well to multiple predictors (in high dimensions, "nearby" observations become sparse — curse of dimensionality).

---

**Q5: How do you choose the number and location of knots in a regression spline?**

Two approaches: (1) **Fixed locations**: place knots at quantiles of X (e.g., 25th, 50th, 75th percentile for 3 knots), distributing them where data is dense. (2) **Select by CV**: try different numbers of knots (equivalently, degrees of freedom) and choose by cross-validation. The number of knots matters more than their exact location — with enough knots at reasonable locations, the spline will adapt to the data. Natural cubic splines with 4–8 degrees of freedom typically suffice for most applications. Smoothing splines avoid this choice entirely by using a knot at every data point and penalizing roughness.

---

**Q6: When would you prefer splines over polynomial regression?**

Almost always. Polynomial regression has several drawbacks: (1) global — a wiggle in one region forces the entire fit to adjust; (2) high-degree polynomials oscillate wildly at the boundaries; (3) numerically unstable for d > 4. Splines are locally adaptive — each region is fit independently (with smoothness constraints at knots). For the same effective degrees of freedom, a natural cubic spline typically outperforms a polynomial. The only advantage of polynomials is simplicity and easy interpretation of coefficients.
