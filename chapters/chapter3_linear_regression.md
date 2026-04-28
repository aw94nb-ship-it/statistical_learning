# Chapter 3: Linear Regression

Linear regression is the foundational supervised learning method for predicting a quantitative response. It is simple, interpretable, and the basis for understanding more complex methods.

---

## Key Concepts

### Simple Linear Regression

Models the relationship between one predictor X and a quantitative response Y:

$$Y \approx \beta_0 + \beta_1 X$$

- $\beta_0$: intercept — expected value of Y when X = 0
- $\beta_1$: slope — average change in Y for a one-unit increase in X

**Least squares estimation**: choose $\hat{\beta}_0$ and $\hat{\beta}_1$ to minimize the **Residual Sum of Squares (RSS)**:

$$\text{RSS} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \sum_{i=1}^{n}(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2$$

The closed-form solution:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}, \qquad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

---

### Assessing Coefficient Accuracy

**Standard error of $\hat{\beta}_1$**:

$$\text{SE}(\hat{\beta}_1)^2 = \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}$$

where $\sigma^2 = \text{Var}(\varepsilon)$, estimated by $\hat{\sigma}^2 = \text{RSS} / (n-2)$.

**t-statistic**: tests $H_0: \beta_1 = 0$ (no relationship between X and Y):

$$t = \frac{\hat{\beta}_1 - 0}{\text{SE}(\hat{\beta}_1)}$$

Under $H_0$ this follows a t-distribution with $n-2$ degrees of freedom. A large |t| → small p-value → reject $H_0$.

**95% confidence interval for $\beta_1$**:

$$\hat{\beta}_1 \pm 2 \cdot \text{SE}(\hat{\beta}_1)$$

---

### Assessing Model Fit

**Residual Standard Error (RSE)**: average amount the response deviates from the regression line.

$$\text{RSE} = \sqrt{\frac{\text{RSS}}{n-2}}$$

Measured in the same units as Y — an absolute measure of fit.

**R² (coefficient of determination)**: proportion of variance in Y explained by X.

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}, \qquad \text{TSS} = \sum_{i=1}^n (y_i - \bar{y})^2$$

- R² ∈ [0, 1]; closer to 1 = better fit
- In simple linear regression, $R^2 = r^2$ (squared correlation between X and Y)
- R² always increases when you add predictors — use **Adjusted R²** for model comparison

$$\text{Adj } R^2 = 1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)}$$

---

### Multiple Linear Regression

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \varepsilon$$

Each $\beta_j$ = average change in Y per unit increase in $X_j$, **holding all other predictors fixed**.

**F-statistic**: tests whether any predictor is useful ($H_0: \beta_1 = \cdots = \beta_p = 0$):

$$F = \frac{(\text{TSS} - \text{RSS})/p}{\text{RSS}/(n-p-1)}$$

Large F → at least one $\beta_j \neq 0$. Always check F before interpreting individual t-statistics — with many predictors, some will be significant by chance.

---

### Qualitative Predictors (Dummy Variables)

For a categorical variable with $k$ levels, create $k-1$ dummy variables. The omitted level is the **baseline**.

Example: `ethnicity` with levels African American, Asian, Caucasian:
- Create `I_Asian` and `I_Caucasian`; African American is baseline
- Coefficient on `I_Asian` = average difference between Asian and African American, all else equal

---

### Interaction Terms

The **additive assumption** says each predictor's effect is independent of the others. Relax it with interactions:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2 + \varepsilon$$

$\beta_3$ = how much the effect of $X_1$ on Y changes as $X_2$ increases.

**Hierarchical principle**: if you include an interaction term, always include the main effects too — even if their p-values are not significant.

---

### Potential Problems

| Problem | Diagnostic | Fix |
|---|---|---|
| **Non-linearity** | Residuals vs. fitted plot: U-shape | Add polynomial terms or transform X/Y |
| **Heteroscedasticity** | Residuals vs. fitted: funnel shape | Transform Y (e.g., log), or use WLS |
| **Correlation of errors** | Residuals vs. time: pattern | Use time series models |
| **Outliers** | Studentized residuals > 3 | Investigate; consider robust regression |
| **High leverage** | Leverage statistic $h_i > (p+1)/n$ | Remove or investigate |
| **Collinearity** | VIF > 5–10 | Drop a variable; ridge regression |

**Variance Inflation Factor (VIF)**: measures how much variance of $\hat{\beta}_j$ is inflated due to collinearity with other predictors.

$$\text{VIF}(\hat{\beta}_j) = \frac{1}{1 - R^2_{X_j | X_{-j}}}$$

VIF = 1: no collinearity. VIF > 10: severe collinearity.

---

### Confidence vs. Prediction Intervals

Both are centered on $\hat{y}$ at a new $x^*$, but they measure different things:

| | What it captures | Width |
|---|---|---|
| **Confidence interval** | Uncertainty about the **mean** response at $x^*$ | Narrower |
| **Prediction interval** | Uncertainty about an **individual** new observation | Wider (adds $\sigma^2$) |

Prediction intervals are always wider because individual observations vary around the mean.

---

## Real Data Example: Advertising Dataset

The Advertising dataset has sales (in thousands of units) and advertising spend across TV, radio, and newspaper for 200 markets.

**Question**: which advertising channels drive sales, and by how much?

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
url = "https://www.statlearning.com/s/Advertising.csv"
ads = pd.read_csv(url, index_col=0)
print(ads.head())
print(ads.describe())

# ── Simple linear regression: TV → Sales ──────────────────────────────────────
model_simple = smf.ols("Sales ~ TV", data=ads).fit()
print(model_simple.summary())

# Key outputs:
# Intercept ≈ 7.03: avg sales with no TV advertising
# TV coef ≈ 0.0475: each $1K increase in TV spend → +47.5 units sold
# R² ≈ 0.61: TV explains 61% of variance in sales

# ── Visualize simple regression ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(ads["TV"], ads["Sales"], alpha=0.6, color="steelblue", label="Data")
x_range = np.linspace(ads["TV"].min(), ads["TV"].max(), 100)
y_pred = model_simple.params["Intercept"] + model_simple.params["TV"] * x_range
ax.plot(x_range, y_pred, color="coral", linewidth=2, label="Fitted line")
ax.set_xlabel("TV Advertising Budget ($K)")
ax.set_ylabel("Sales (thousands of units)")
ax.set_title("Simple Linear Regression: TV → Sales")
ax.legend()
plt.tight_layout()
plt.show()

# ── Multiple linear regression: TV + Radio + Newspaper → Sales ────────────────
model_multi = smf.ols("Sales ~ TV + Radio + Newspaper", data=ads).fit()
print(model_multi.summary())

# Key outputs:
# TV coef ≈ 0.046 (p < 0.001): TV has a significant positive effect
# Radio coef ≈ 0.189 (p < 0.001): Radio has the largest effect per dollar
# Newspaper coef ≈ 0.001 (p = 0.86): Newspaper is NOT significant
# R² ≈ 0.897: model explains 90% of variance

# ── Model with interaction: TV × Radio ────────────────────────────────────────
model_interact = smf.ols("Sales ~ TV + Radio + TV:Radio", data=ads).fit()
print(model_interact.summary())
# TV:Radio interaction is significant → synergy effect
# Spending on both TV and Radio together has a greater effect than the sum of parts

# ── Residual diagnostics ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residuals vs. fitted
fitted = model_multi.fittedvalues
resid = model_multi.resid
axes[0].scatter(fitted, resid, alpha=0.5, color="steelblue")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Fitted values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs. Fitted")

# Q-Q plot
from scipy import stats
(osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
axes[1].scatter(osm, osr, alpha=0.5, color="steelblue")
axes[1].plot(osm, slope * np.array(osm) + intercept, color="red", linewidth=2)
axes[1].set_xlabel("Theoretical quantiles")
axes[1].set_ylabel("Sample quantiles")
axes[1].set_title("Normal Q-Q Plot")

plt.tight_layout()
plt.show()

# ── VIF: check for collinearity ───────────────────────────────────────────────
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = ads[["TV", "Radio", "Newspaper"]]
X_with_const = pd.concat([pd.Series(1, index=X.index, name="const"), X], axis=1)
vif = pd.DataFrame({
    "feature": X_with_const.columns[1:],
    "VIF": [variance_inflation_factor(X_with_const.values, i+1)
            for i in range(len(X.columns))]
})
print(vif)
# TV VIF ≈ 1.0, Radio ≈ 1.1, Newspaper ≈ 1.1 → no collinearity problem

# ── Confidence vs. prediction intervals ──────────────────────────────────────
new_data = pd.DataFrame({"TV": [100], "Radio": [20], "Newspaper": [10]})
pred = model_multi.get_prediction(new_data)
print("Confidence interval (mean response):")
print(pred.summary_frame(alpha=0.05)[["mean", "mean_ci_lower", "mean_ci_upper"]])
print("Prediction interval (individual response):")
print(pred.summary_frame(alpha=0.05)[["mean", "obs_ci_lower", "obs_ci_upper"]])
```

---

## Interview Questions

**Q1: What is the interpretation of a regression coefficient?**

The coefficient $\hat{\beta}_j$ is the average change in Y for a one-unit increase in $X_j$, **holding all other predictors constant**. This "all else equal" qualifier is critical — it only holds within the observed data range and assumes the model is correctly specified.

---

**Q2: What does R² tell you, and what doesn't it tell you?**

R² measures the proportion of variance in Y explained by the model — how well the predictors collectively fit the data. What it doesn't tell you: (1) whether the model is correctly specified (non-linear relationships can have high R²); (2) whether individual coefficients are significant; (3) whether the model will generalize to new data; (4) causality — a high R² just means good prediction, not that X causes Y. A model with many predictors will always have a high R² even if the predictors are noise — use Adjusted R² for model comparison.

---

**Q3: What is the difference between a confidence interval and a prediction interval?**

A confidence interval for $\hat{y}$ at $x^*$ captures uncertainty about the **mean response** — how much the true average of Y might vary given sampling variability in the coefficients. A prediction interval captures uncertainty about an **individual new observation** — it adds the irreducible variance $\sigma^2$ on top of coefficient uncertainty. Prediction intervals are always wider. In practice: use CI when you want to estimate the average behavior; use PI when you want to predict a specific new value.

---

**Q4: What are the assumptions of linear regression and how do you check them?**

1. **Linearity**: relationship between X and Y is linear → check residuals vs. fitted plot (should be random, no pattern)
2. **Independence**: errors are uncorrelated → check residuals vs. time/order for patterns
3. **Homoscedasticity**: constant error variance → check residuals vs. fitted for funnel shape
4. **Normality of errors**: errors are normally distributed → check Q-Q plot of residuals
5. **No perfect multicollinearity**: predictors aren't perfectly correlated → check VIF

The most important in practice are linearity and homoscedasticity. Normality matters mainly for inference in small samples; with large n, CLT makes the normality assumption less critical.

---

**Q5: How do you detect and handle multicollinearity?**

Detection: Variance Inflation Factor (VIF). VIF > 5 is a warning; VIF > 10 is severe. Multicollinearity inflates standard errors — individual coefficients become unreliable even if the overall model fits well.

Fixes: (1) drop one of the correlated predictors; (2) combine them (PCA); (3) use ridge regression, which adds an L2 penalty that shrinks correlated coefficients together; (4) collect more data (increases denominator in SE formula). The key point: multicollinearity doesn't affect prediction quality, only interpretation of individual coefficients.

---

**Q6: When should you use linear regression vs. more complex models?**

Linear regression is preferable when: (1) the relationship is approximately linear; (2) interpretability is required (coefficient = business meaning); (3) n is small relative to p (complex models overfit); (4) you need confidence/prediction intervals with valid coverage.

Use complex models when: non-linearity is evident in residuals; interactions are complex and hard to specify manually; predictive accuracy matters more than interpretability; n is large enough to support complexity.

---

**Q7: What is the F-statistic and when do you use it?**

The F-statistic tests whether at least one predictor is useful: $H_0: \beta_1 = \cdots = \beta_p = 0$. With many predictors, some individual t-tests will be significant by chance (multiple testing problem). The F-test controls for this — if F is not significant, don't trust individual t-statistics even if some look significant. Always check F before interpreting individual coefficients in a multiple regression.
