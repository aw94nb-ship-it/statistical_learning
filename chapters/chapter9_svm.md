# Chapter 9: Support Vector Machines

SVMs are a class of classifiers based on the concept of a **decision boundary** (hyperplane) that separates classes with maximum margin. They can handle linear and non-linear classification via the **kernel trick**.

---

## Key Concepts

### The Maximal Margin Classifier

For a linearly separable binary classification problem, infinitely many hyperplanes separate the two classes. The **maximal margin classifier** chooses the hyperplane that **maximizes the margin** — the smallest perpendicular distance from any training observation to the decision boundary.

A hyperplane in p dimensions: $\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p = 0$

**Optimization**:
$$\max_{\beta_0, \beta_1, \ldots, \beta_p, M} M \quad \text{subject to:} \quad y_i(\beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}) \geq M$$

where M is the margin width and $\|\beta\| = 1$.

**Support vectors**: the observations that lie exactly on the margin boundary. The classifier depends **only** on the support vectors — not on all training observations. This makes it memory-efficient but also sensitive to the positions of support vectors.

**Limitation**: requires linear separability. Breaks down completely when classes overlap.

---

### The Support Vector Classifier (Soft Margin)

Allows some observations to be on the wrong side of the margin (or even the hyperplane) using **slack variables** $\epsilon_i \geq 0$:

$$\max M \quad \text{subject to:} \quad y_i(\beta_0 + \beta_i^T x_i) \geq M(1 - \epsilon_i), \quad \sum_{i=1}^n \epsilon_i \leq C, \quad \epsilon_i \geq 0$$

- $\epsilon_i = 0$: observation is on the correct side of the margin
- $0 < \epsilon_i < 1$: observation is within the margin but on the correct side
- $\epsilon_i > 1$: observation is misclassified

**C** is the tuning parameter (budget for total margin violations):
- **C = 0**: no violations allowed → maximal margin classifier (may not exist if not separable)
- **Large C**: many violations allowed → wider margin, more regularized, lower variance
- **Small C**: few violations → narrow margin, fits training data more tightly, higher variance

Choose C by cross-validation.

---

### Support Vector Machines (Non-linear Classification)

What if the decision boundary is non-linear? The SVM **enlarges the feature space** using transformations of the predictors, then finds a linear boundary in the enlarged space (which is non-linear in the original space).

**Key insight**: the classifier can be written in terms of **inner products** $\langle x_i, x_{i'} \rangle = \sum_j x_{ij} x_{i'j}$ between observations. To use non-linear features, replace the inner product with a **kernel function** $K(x_i, x_{i'})$.

**Common kernels**:

| Kernel | Formula | Decision boundary |
|---|---|---|
| Linear | $K(x_i, x_{i'}) = \sum_j x_{ij} x_{i'j}$ | Linear hyperplane |
| Polynomial | $K(x_i, x_{i'}) = (1 + \sum_j x_{ij} x_{i'j})^d$ | Polynomial of degree d |
| Radial (RBF) | $K(x_i, x_{i'}) = \exp\left(-\gamma \sum_j (x_{ij} - x_{i'j})^2\right)$ | Highly non-linear, local |
| Sigmoid | $K(x_i, x_{i'}) = \tanh(\gamma_1 \sum_j x_{ij} x_{i'j} + \gamma_2)$ | Neural network-like |

**The kernel trick**: using $K(x_i, x_{i'})$ is computationally equivalent to computing inner products in the enlarged feature space — but we never have to explicitly compute the (possibly infinite-dimensional) feature space.

**Radial kernel**: observation $x_i$ has high influence on the classification of $x$ only when $x_i$ is close to $x$ (because the kernel value decays exponentially with distance). This gives a highly local decision boundary.

**Tuning parameters**:
- **C** (cost): controls the bias-variance tradeoff (same as soft-margin SVM)
- **Kernel parameters**: degree d (polynomial), γ (radial/RBF)
- Choose all by cross-validation

---

### SVM for More Than Two Classes

SVM is inherently binary. Two extensions:

1. **One-versus-one (OVO)**: fit $\binom{K}{2}$ classifiers, each distinguishing one pair of classes. Assign observation to the class that wins the most pairwise comparisons.

2. **One-versus-rest (OVR)**: fit K classifiers, each distinguishing class k vs. all others. Assign to class k with the largest decision value.

OVO is more common in practice and handles class imbalance better.

---

### SVMs vs. Logistic Regression

Both find a linear decision boundary (in the original or kernel-mapped space). Key differences:

| | SVM | Logistic Regression |
|---|---|---|
| **Objective** | Maximize margin | Maximize likelihood |
| **Loss function** | Hinge loss: max(0, 1 − y·f(x)) | Log-loss (cross-entropy) |
| **Sparsity** | Only support vectors matter | All observations contribute |
| **Probabilities** | Not naturally produced | Outputs calibrated probabilities |
| **Non-linearity** | Kernel trick | Manual feature engineering |
| **Scaling** | Required | Required |

**When SVMs win**: when the data is nearly linearly separable; for text classification (high-dimensional, sparse); when a clear margin exists. **When logistic regression wins**: when you need calibrated probabilities; for imbalanced classes; when interpretability matters; when n is much larger than p.

---

### SVMs for Regression (SVR)

Support Vector Regression uses a similar framework for continuous outcomes, with an **ε-insensitive tube**: errors smaller than ε are ignored; only observations outside the tube contribute to the loss.

$$\min \frac{1}{2}\|\beta\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)$$

subject to $y_i - f(x_i) \leq \varepsilon + \xi_i$, $f(x_i) - y_i \leq \varepsilon + \xi_i^*$, $\xi_i, \xi_i^* \geq 0$.

---

## Real Data Example: Heart Disease Classification

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      GridSearchCV, KFold)
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import Pipeline

# ── Load Heart disease dataset ────────────────────────────────────────────────
try:
    from ISLP import load_data
    Heart = load_data("Heart")
except ImportError:
    url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Heart.csv"
    Heart = pd.read_csv(url, index_col=0)

Heart = Heart.dropna()
Heart["AHD_bin"] = (Heart["AHD"] == "Yes").astype(int)

# Encode categorical features
Heart = pd.get_dummies(Heart, columns=["ChestPain", "Thal"], drop_first=True)
feature_cols = [c for c in Heart.columns if c not in ["AHD", "AHD_bin"]]
X = Heart[feature_cols].astype(float).values
y = Heart["AHD_bin"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                      random_state=42, stratify=y)

# ── 2D visualization: use only Age and MaxHR for plotting ─────────────────────
X2 = Heart[["Age", "MaxHR"]].values
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.3, random_state=42, stratify=y)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
kernels = [("linear", {}), ("poly", {"degree": 3}), ("rbf", {})]

for ax, (kern, params) in zip(axes, kernels):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svc", SVC(kernel=kern, C=1, **params))])
    pipe.fit(X2_train, y2_train)
    DecisionBoundaryDisplay.from_estimator(pipe, X2_train, ax=ax,
                                            alpha=0.3, cmap="RdBu")
    ax.scatter(X2_train[:, 0], X2_train[:, 1], c=y2_train,
               cmap="RdBu", edgecolors="k", s=30, alpha=0.8)
    ax.set_title(f"SVM (kernel={kern})")
    ax.set_xlabel("Age")
    ax.set_ylabel("Max HR")

plt.tight_layout()
plt.show()

# ── Tuning C and gamma via GridSearchCV ───────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

param_grid = {
    "C":     [0.01, 0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1, "scale"]
}

gs = GridSearchCV(SVC(kernel="rbf", probability=True), param_grid,
                  cv=5, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train_s, y_train)

print(f"Best params: {gs.best_params_}")
print(f"Best CV AUC: {gs.best_score_:.4f}")

svm_best = gs.best_estimator_
y_prob = svm_best.predict_proba(X_test_s)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"Test AUC: {auc:.4f}")
print(classification_report(y_test, svm_best.predict(X_test_s)))

# ── Compare kernels ───────────────────────────────────────────────────────────
print("\nKernel comparison (AUC, 5-fold CV):")
for kern in ["linear", "poly", "rbf"]:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svc", SVC(kernel=kern, probability=True))])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"  {kern:8s}: {scores.mean():.4f} ± {scores.std():.4f}")

# ── ROC curve: SVM vs Logistic Regression ────────────────────────────────────
from sklearn.linear_model import LogisticRegression

lr = Pipeline([("scaler", StandardScaler()),
               ("lr", LogisticRegression(max_iter=1000))])
lr.fit(X_train, y_train)
lr_prob = lr.predict_proba(X_test)[:, 1]

fig, ax = plt.subplots(figsize=(7, 5))
for probs, label, color in [(y_prob, "SVM (RBF)", "coral"),
                              (lr_prob, "Logistic Regression", "steelblue")]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_ = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC={auc_:.3f})")

ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("SVM vs Logistic Regression: Heart Disease")
ax.legend()
plt.tight_layout()
plt.show()

# ── Effect of C on number of support vectors ─────────────────────────────────
print("\nEffect of C (RBF SVM):")
for C in [0.01, 0.1, 1, 10, 100]:
    svm = SVC(kernel="rbf", C=C)
    svm.fit(X_train_s, y_train)
    train_acc = svm.score(X_train_s, y_train)
    test_acc  = svm.score(X_test_s, y_test)
    n_sv = svm.n_support_.sum()
    print(f"  C={C:6.2f}: train={train_acc:.3f}, test={test_acc:.3f}, "
          f"support vectors={n_sv}")
```

---

## Interview Questions

**Q1: What is the margin in an SVM and why do we want to maximize it?**

The margin is the width of the "band" on either side of the decision hyperplane, measured as twice the distance from the hyperplane to the nearest training observation. Maximizing the margin is a form of regularization: a larger margin means the decision boundary is as far as possible from all training points, giving more room for new observations to fall on the correct side. Intuitively, a boundary that barely separates training observations is overfit to noise; a boundary with a large margin generalizes better. The observations on the margin boundary are the support vectors — the classifier depends only on them.

---

**Q2: What is the kernel trick in SVMs, and why is it useful?**

The SVM decision function can be written entirely in terms of inner products $\langle x_i, x_{i'} \rangle$ between observations. To get a non-linear boundary, we could transform features to a higher-dimensional space and compute inner products there — but this may be computationally expensive or the space may be infinite-dimensional. The kernel trick: replace $\langle x_i, x_{i'} \rangle$ with a kernel function $K(x_i, x_{i'})$ that implicitly computes inner products in the transformed space without explicitly constructing it. The RBF kernel $K(x_i, x_{i'}) = \exp(-\gamma \|x_i - x_{i'}\|^2)$ corresponds to an infinite-dimensional feature space, making it highly flexible.

---

**Q3: What does the cost parameter C control in SVM?**

C is the budget for margin violations. Small C: many violations allowed → wider, more regularized margin → fewer support vectors → high bias, low variance. Large C: few violations allowed → narrow margin, tighter fit to training data → more support vectors → low bias, high variance (can overfit). As C → ∞, SVM becomes the maximal margin classifier (no violations at all). Choose C by cross-validation. With an RBF kernel, C and γ interact: large C + large γ → overfit; small C + small γ → underfit. Always tune both together via grid search.

---

**Q4: When would you use an RBF kernel vs. a linear kernel?**

Use a linear kernel when: (1) the feature space is already very high-dimensional (e.g., text with TF-IDF features) — non-linear kernels are unlikely to help; (2) n << p — the linear boundary is already complex enough; (3) you need interpretability. Use an RBF kernel when: (1) the decision boundary is clearly non-linear; (2) moderate p and n; (3) after a linear kernel fails (check training vs. validation accuracy). In practice, try linear first — if it performs well, stop. If not, try RBF with CV over C and γ.

---

**Q5: How does SVM compare to logistic regression?**

Both find linear decision boundaries (in the original or kernel space). SVM maximizes margin (hinge loss); logistic regression maximizes likelihood (log-loss). Key practical differences: (1) SVM doesn't naturally output calibrated probabilities (needs Platt scaling — fits a logistic model on the SVM outputs); (2) SVM depends only on support vectors → more memory-efficient but less interpretable; (3) SVM handles non-linearity via kernels; logistic regression requires manual feature engineering; (4) both require feature scaling. When to prefer logistic regression: need probabilities, class imbalance, interpretability, very large n (SVM training is O(n²)–O(n³)). When to prefer SVM: clear margin, high-dimensional sparse data, text classification.

---

**Q6: What is a support vector, and how does the SVM depend on it?**

A support vector is a training observation that lies on or within the margin (including misclassified observations in the soft-margin SVM). These are the only observations that influence the decision boundary — all other observations could be moved without changing the classifier. The number of support vectors relates inversely to C: small C → many support vectors (relaxed margin); large C → few support vectors (tight margin). This makes SVM memory-efficient in production but also means it is sensitive to the positions of the support vectors — outliers on the margin can substantially change the boundary.
