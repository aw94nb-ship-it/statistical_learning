# Chapter 4: Classification — Logistic Regression

Classification predicts a qualitative (categorical) response. The most important method is **logistic regression**, which models the probability that Y belongs to a particular class.

---

## Key Concepts

### Why Not Linear Regression for Classification?

Linear regression can produce probabilities outside [0, 1] and assumes a linear relationship between X and P(Y=1), which is rarely appropriate. With more than two classes, the arbitrary ordering of categories (1, 2, 3) implies a distance structure that doesn't exist.

**The fix**: model the probability directly using a function that maps any real number to [0, 1].

---

### The Logistic Model

$$P(Y=1 \mid X) = p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}$$

This is the **logistic function** (sigmoid), which always outputs a value in (0, 1).

**Log-odds (logit)**:

$$\log\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1 X$$

The log-odds are linear in X, even though the probability is not.

**Interpretation of $\beta_1$**: a one-unit increase in X increases the log-odds of Y=1 by $\beta_1$. Equivalently, it multiplies the odds by $e^{\beta_1}$.

- $\beta_1 > 0$: increasing X increases P(Y=1)
- $\beta_1 < 0$: increasing X decreases P(Y=1)
- $e^{\beta_1}$ is the **odds ratio**: how much the odds change per unit increase in X

---

### Estimation: Maximum Likelihood

Unlike linear regression (which uses least squares), logistic regression uses **maximum likelihood estimation (MLE)**. We choose $\beta_0, \beta_1$ to maximize the likelihood:

$$\ell(\beta_0, \beta_1) = \prod_{i: y_i=1} p(x_i) \prod_{i: y_i=0} (1 - p(x_i))$$

Or equivalently, minimize the **log-loss (binary cross-entropy)**:

$$-\ell = -\sum_{i=1}^n \left[y_i \log(p(x_i)) + (1-y_i)\log(1-p(x_i))\right]$$

No closed form — solved by numerical optimization (Newton-Raphson / gradient descent).

---

### Multiple Logistic Regression

$$\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

Each $\beta_j$ = change in log-odds per unit increase in $X_j$, holding others fixed.

---

### Multinomial Logistic Regression

For K > 2 classes, model K−1 log-odds relative to a baseline class K:

$$\log\left(\frac{P(Y=k \mid X)}{P(Y=K \mid X)}\right) = \beta_{k0} + \beta_{k1} X_1 + \cdots + \beta_{kp} X_p$$

**K vs k:**

| Symbol | Meaning |
|---|---|
| **K** (uppercase) | Total number of classes AND the label for the baseline class |
| **k** (lowercase) | A specific class being modeled, where k = 1, 2, ..., K−1 |

You fit **K−1 equations** — one for each non-baseline class compared against K. Each equation has its own intercept and slopes.

**Example: 3 classes (K = 3) — Bus, Car, Train. Baseline = Train.**

```
Equation 1 (k = Bus):  log[ P(Bus) / P(Train) ] = β₁₀ + β₁₁·distance
Equation 2 (k = Car):  log[ P(Car) / P(Train) ] = β₂₀ + β₂₁·distance
```

Each equation has its own intercept and own slopes — distance might push strongly toward Car but weakly toward Bus. Train gets no equation — its probability is whatever is left over.

**Why K−1, not K equations?** If you fit all K, the system is over-identified — you can add a constant to every equation and get identical probabilities. Fixing one class as baseline forces a unique solution. The choice of baseline doesn't affect predictions, only how coefficients are interpreted.

**Joint estimation:** the K−1 equations are estimated simultaneously by maximizing one combined likelihood — not independently. This ensures all predicted probabilities stay coherent (sum to 1).

**Softmax** converts the 2 equations back to probabilities for all 3 classes:

$$P(\text{Bus}) = \frac{e^{\beta_{10} + \beta_{11}X}}{1 + e^{\beta_{10} + \beta_{11}X} + e^{\beta_{20} + \beta_{21}X}}$$

$$P(\text{Car}) = \frac{e^{\beta_{20} + \beta_{21}X}}{1 + e^{\beta_{10} + \beta_{11}X} + e^{\beta_{20} + \beta_{21}X}}$$

$$P(\text{Train}) = \frac{1}{1 + e^{\beta_{10} + \beta_{11}X} + e^{\beta_{20} + \beta_{21}X}}$$

All three sum to 1. Train's numerator is always 1 (= e⁰) because it's the baseline.

**General softmax** converts the K−1 log-odds back to probabilities for all K classes:

$$P(Y=k \mid X) = \frac{e^{\beta_{k0} + \beta_{k1}X_1 + \cdots + \beta_{kp}X_p}}{1 + \sum_{l=1}^{K-1} e^{\beta_{l0} + \beta_{l1}X_1 + \cdots + \beta_{lp}X_p}}$$

The baseline class K always has numerator = 1 (= e⁰). All K probabilities sum to 1.

---

### Model Assessment

**Confusion Matrix** (for a binary classifier at threshold 0.5):

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

Key metrics:
- **Accuracy** = (TP + TN) / n — misleading when classes are imbalanced
- **Sensitivity (Recall, TPR)** = TP / (TP + FN) — fraction of true positives caught
- **Specificity (TNR)** = TN / (TN + FP) — fraction of true negatives caught
- **Precision** = TP / (TP + FP) — fraction of predicted positives that are correct
- **F1** = 2 × (Precision × Recall) / (Precision + Recall)

**ROC Curve and AUC**

A logistic regression outputs a **probability**, not a class. You pick a threshold to convert it to 0/1 — and every threshold gives a different TPR/FPR pair. The ROC curve plots all of them at once by sweeping the threshold from 1 down to 0.

*Example: 10 patients, 5 sick (y=1), 5 healthy (y=0)*

| Patient | True label | p(sick) |
|---|---|---|
| A | 1 | 0.95 |
| B | 1 | 0.80 |
| C | 0 | 0.70 |
| D | 1 | 0.60 |
| E | 0 | 0.55 |
| F | 1 | 0.45 |
| G | 0 | 0.40 |
| H | 0 | 0.30 |
| I | 1 | 0.20 |
| J | 0 | 0.10 |

| Threshold | Predict positive | TP | FP | TPR | FPR |
|---|---|---|---|---|---|
| > 0.95 | nobody | 0 | 0 | 0.0 | 0.0 |
| > 0.70 | A, B | 2 | 0 | 0.4 | 0.0 |
| > 0.55 | A, B, C, D | 3 | 1 | 0.6 | 0.2 |
| > 0.40 | A–F | 4 | 2 | 0.8 | 0.4 |
| > 0.10 | A–I | 5 | 4 | 1.0 | 0.8 |
| > 0.00 | everyone | 5 | 5 | 1.0 | 1.0 |

Plot these (FPR, TPR) points → that's your ROC curve.

```
TPR
1.0 |          * ----*
    |       *
0.5 |     *
    |   *
0.0 *-----------*----
    0.0  0.5   1.0  FPR

    ↑ Good model: hugs top-left corner
    ↑ Random model: diagonal line (AUC = 0.5)
```

**AUC (Area Under the ROC Curve)**: probability that the model scores a random positive higher than a random negative.

| AUC | Meaning |
|---|---|
| 1.0 | Perfect — every positive ranks above every negative |
| 0.9 | Model ranks a random positive above a random negative 90% of the time |
| 0.5 | Random — coin flip |
| < 0.5 | Worse than random (flip your predictions) |

**AUC vs accuracy — when each matters**

| Use AUC when | Use accuracy when |
|---|---|
| Classes are imbalanced | Classes are balanced |
| You care about ranking, not a fixed threshold | You have a fixed operating threshold |
| Comparing models before choosing a threshold | Final deployed model evaluation |

---

### Choosing a Threshold

The default threshold (0.5) is not always optimal:
- **High-stakes false negatives** (missed cancer): lower the threshold → increase sensitivity at cost of specificity
- **High-stakes false positives** (fraud alerts): raise the threshold → increase precision at cost of recall

The ROC curve shows the tradeoff at all thresholds — choose based on the cost structure.

---

### Linear Discriminant Analysis (LDA)

LDA models the distribution of X separately in each class, then uses Bayes' theorem to compute P(Y=k|X).

**Key assumptions**:
- X is normally distributed within each class
- All classes share the **same covariance matrix** Σ

**LDA decision boundary** is linear in X.

**When to prefer LDA over logistic regression**:
- Classes are well-separated (logistic regression can be unstable)
- n is small and the normality assumption is reasonable
- More than 2 classes (LDA handles K classes naturally)

### Quadratic Discriminant Analysis (QDA)

Same as LDA but **each class has its own covariance matrix**. Decision boundary is quadratic. More flexible than LDA but requires more parameters — prefer when n is large relative to p.

### Naive Bayes

Assumes features are **conditionally independent** given class. Despite this unrealistic assumption, works well in practice for high-dimensional problems (text classification, spam detection).

---

### Comparison of Methods

| Method | Decision Boundary | Assumptions | Best When |
|---|---|---|---|
| Logistic Regression | Linear | None on X | n large, features numeric |
| LDA | Linear | Normal X, equal Σ | Small n, classes well-separated |
| QDA | Quadratic | Normal X, unequal Σ | Moderate n, non-linear boundary |
| Naive Bayes | Flexible | Feature independence | High-dimensional, mixed features |
| KNN | Non-parametric | None | Non-linear boundary, large n |

---

## Real Data Example: Default Dataset

The Default dataset has 10,000 observations. Goal: predict whether a credit card holder will default (`default = Yes/No`) based on `balance`, `income`, and `student` status.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import statsmodels.formula.api as smf

# Load data (from ISLP package or manual)
try:
    from ISLP import load_data
    Default = load_data("Default")
except ImportError:
    url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Default.csv"
    Default = pd.read_csv(url)

Default["default_bin"] = (Default["default"] == "Yes").astype(int)
Default["student_bin"] = (Default["student"] == "Yes").astype(int)

print(Default.head())
print(f"\nDefault rate: {Default['default_bin'].mean():.3f}")

# ── Logistic regression: balance → default ────────────────────────────────────
model_simple = smf.logit("default_bin ~ balance", data=Default).fit()
print(model_simple.summary())

# Coefficient on balance ≈ 0.0055: one dollar increase in balance
# increases log-odds of default by 0.0055
# Odds ratio: exp(0.0055) ≈ 1.0055 → 0.55% increase in odds per dollar

# ── Visualize probability curve ───────────────────────────────────────────────
balance_range = np.linspace(Default["balance"].min(), Default["balance"].max(), 300)
log_odds = model_simple.params["Intercept"] + model_simple.params["balance"] * balance_range
prob = 1 / (1 + np.exp(-log_odds))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(Default["balance"], Default["default_bin"],
                alpha=0.05, color="steelblue", label="Observations")
axes[0].plot(balance_range, prob, color="coral", linewidth=2, label="P(default=Yes)")
axes[0].set_xlabel("Balance ($)")
axes[0].set_ylabel("Probability of Default")
axes[0].set_title("Logistic Regression: Balance → Default")
axes[0].legend()

# ── Multiple logistic regression ──────────────────────────────────────────────
model_multi = smf.logit("default_bin ~ balance + income + student_bin", data=Default).fit()
print(model_multi.summary())

# Note: student coefficient is NEGATIVE in multiple regression
# even though students have higher default rate in simple regression
# This is because students carry higher balances → confounding

# ── Train/test split and evaluation ──────────────────────────────────────────
X = Default[["balance", "income", "student_bin"]]
y = Default["default_bin"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                      random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
disp.plot(ax=axes[1], colorbar=False)
axes[1].set_title("Confusion Matrix (threshold = 0.5)")
plt.tight_layout()
plt.show()

print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

# ── ROC curve and AUC ─────────────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="coral", linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random classifier")
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC Curve: Default Prediction")
ax.legend()
plt.tight_layout()
plt.show()

# ── Effect of threshold on sensitivity/specificity ───────────────────────────
print("\nThreshold sensitivity analysis:")
print(f"{'Threshold':>10} {'Sensitivity':>12} {'Specificity':>12} {'Precision':>10}")
for thresh in [0.1, 0.2, 0.3, 0.5, 0.7]:
    y_pred_t = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"{thresh:>10.1f} {sens:>12.3f} {spec:>12.3f} {prec:>10.3f}")

# ── LDA comparison ────────────────────────────────────────────────────────────
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_auc = roc_auc_score(y_test, lda.predict_proba(X_test)[:, 1])

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
qda_auc = roc_auc_score(y_test, qda.predict_proba(X_test)[:, 1])

print(f"\nAUC comparison:")
print(f"  Logistic Regression: {auc:.3f}")
print(f"  LDA:                 {lda_auc:.3f}")
print(f"  QDA:                 {qda_auc:.3f}")
```

---

## Interview Questions

**Q1: Why can't you use linear regression for a binary classification problem?**

Linear regression can produce predicted probabilities outside [0, 1], which are not valid probabilities. The linear model also assumes the effect of X on P(Y=1) is constant across all values of X, but in reality the effect tapers off near 0 and 1. With more than two classes, the arbitrary numeric coding (1, 2, 3) implies an ordering and equal spacing that doesn't exist for nominal categories.

---

**Q2: What is the interpretation of a logistic regression coefficient?**

The coefficient $\hat{\beta}_j$ is the change in **log-odds** of Y=1 per one-unit increase in $X_j$, holding other predictors constant. More intuitively: $e^{\hat{\beta}_j}$ is the **odds ratio** — how much the odds of Y=1 multiply per unit increase in $X_j$. Note: this is not the change in probability, which depends on the current value of $X_j$ (the probability curve is S-shaped, not linear).

---

**Q3: What is AUC and how do you interpret it?**

AUC (Area Under the ROC Curve) is the probability that the model assigns a higher predicted probability to a randomly chosen positive case than to a randomly chosen negative case. AUC = 0.5 → random; AUC = 1.0 → perfect separation. AUC is threshold-independent — it summarizes model performance across all possible classification thresholds. It's particularly useful when classes are imbalanced and the choice of threshold matters.

---

**Q4: When would you lower the classification threshold below 0.5?**

When the cost of false negatives (missing a positive) is higher than the cost of false positives. Examples: cancer screening (missing a case is worse than an unnecessary biopsy), fraud detection (missing fraud is worse than an unnecessary review), loan default prediction (failing to catch a default may outweigh the cost of rejecting a good customer). Lowering the threshold increases sensitivity at the cost of specificity. The right threshold depends on the cost structure, not just model accuracy.

---

**Q5: What is the difference between sensitivity and precision, and when does each matter?**

- **Sensitivity (recall)** = TP / (TP + FN): of all actual positives, what fraction did we catch? Important when missing positives is costly (disease screening, fraud).
- **Precision** = TP / (TP + FP): of all predicted positives, what fraction are correct? Important when false alarms are costly (spam filters, recommendation systems — annoying the user).

They trade off against each other. The F1 score balances both. In imbalanced datasets, accuracy is misleading — always look at precision and recall separately.

---

**Q6: When would you prefer LDA over logistic regression?**

LDA is preferred when: (1) classes are well-separated — logistic regression can give unstable coefficient estimates when separation is perfect (complete separation problem); (2) the normality assumption is reasonable — LDA is more efficient when it holds; (3) you have more than two classes — LDA handles K classes naturally with a single model; (4) n is small relative to p — LDA's parametric assumptions give it better variance-bias tradeoff in small samples. Logistic regression is generally more robust when the normality assumption fails.

---

**Q7: What is the confounding problem in the Default student example?**

In simple logistic regression, `student` has a positive coefficient — students default more. But in multiple logistic regression with `balance` included, `student` has a negative coefficient. This is confounding: students tend to carry higher balances, and balance is the true driver of default. Within any balance level, students actually default less (possibly better credit behavior or financial discipline). Without controlling for balance, the student variable picks up the effect of balance — a classic omitted variable bias / confounding situation.
