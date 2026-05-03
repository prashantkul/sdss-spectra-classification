# 03 — Why misclassifications happen (from first principles)

---

## 1. The irreducible core: overlapping class-conditional distributions

We observe \(x\) and wish to infer \(y\). The **Bayes error rate** is the lowest achievable average error **for any** classifier under the true \(P(x, y)\). It is **strictly positive** whenever, on a set of \(x\) with positive measure, **more than one** class has non-negligible posterior mass:

\[
P(y_1 \mid x) \approx P(y_2 \mid x) \implies \text{no decision is simultaneously correct for both with probability 1.}
\]

Equivalently: if the **class-conditional densities** \(p(x \mid y=\text{QSO})\) and \(p(x \mid y=\text{GALAXY})\) **overlap** in magnitude–colour–redshift space, **some** fraction of objects will be mislabelled by **any** model using only \(x\).

**Implication for this project:** Photometry-only inputs guarantee **non-zero** Bayes error along the **QSO–galaxy** boundary; misclassifications are expected **in principle**, not an accident of Random Forests.

---

## 2. Measurement noise moves points across boundaries

Real \(x\) is corrupted: photometric errors, calibration drift, and extinction residuals make each object a **sample** from \(p(x \mid y)\), not a fixed point. A **small perturbation** \(\delta x\) can flip the **argmax** of \(P(y \mid x)\) near the decision surface.

**Implication:** Errors **cluster** in regions where class manifolds **approach** in feature space—exactly what colour–redshift scatter plots in the notebook highlight.

---

## 3. Labels are noisy: error in \(y\), not only in \(\hat{y}\)

Training assumes \(y_i\) is **ground truth**. In practice, spectroscopic classification is a **pipeline output**: failures in redshift, line fitting, blends, or composite systems mean a **nonzero** fraction of rows have **wrong or ambiguous** \(y\).

The learner minimizes loss against those labels; it **absorbs** label noise as if it were feature noise. Some “errors” on the test set are **disagreements between the model and a noisy supervisor**, not failures to learn a clean rule.

**Implication:** Perfect accuracy is **not** the theoretical ceiling; the ceiling is **1 − (Bayes error + label noise rate)** in the loose sense.

---

## 4. What the fitted model does: partition feature space

A Random Forest induces a **piecewise-constant** estimate of the posterior (via voting over leaves). **Misclassifications** occur when the **majority vote** in the leaf (or across trees) picks the wrong mode of \(P(y \mid x)\).

**Dominant confusion pair (e.g. QSO → GALAXY):** means the partition places **many** true QSOs in regions where **galaxy** votes win—because **training examples** in those regions were mostly labelled galaxy, or because **overlap** makes both classes locally plausible and **priors + sampling** tip the vote.

---

## 5. Confident wrong predictions

`predict_proba` from trees reflects **leaf occupancy and tree agreement**. **High confidence + wrong class** means: all trees largely agree on a region that **disagrees** with the spectroscopic label for that object.

**First-principle interpretations (not mutually exclusive):**

1. **Sharp partition** near **true** Bayes boundary—object sits on the “wrong” side of an inevitable ambiguity.  
2. **Overconfident** probability estimate (miscalibration).  
3. **Label error** or outlier \(x\).

**Implication:** These points are the **highest information** for error analysis and for deciding whether to add features or clean labels.

---

## 6. Astrophysical intuition as a *consequence*, not an axiom

Unresolved quasars and compact galaxies share **similar photometric signatures** when morphology is weakly encoded in \(x\). Dust reddening **shears** colours, enlarging overlap. These are **mechanisms that widen \(p(x \mid y)\) overlap**—they **explain** the statistical picture above in physical language, not a separate law of “why ML fails.”

---

## 7. What would actually shrink error?

Only moves that **change \(x\)** (new observables), **change \(P(y)\)** handling (priors, costs), or **improve labels** can shift the fundamental tradeoff. Switching from RF to another learner on the **same** \(x\) can only approach the **same** Bayes limit more efficiently—it cannot remove overlap that is **intrinsic** to photometry alone.

---

*Empirical confusion pairs and plots: `sdss_analysis.ipynb` §11c and `docs/plots/misclassification-analysis.png`.*
