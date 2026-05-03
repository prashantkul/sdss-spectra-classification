# 02 — Future work (from first principles)

Each item below starts from a **gap between the ideal target** (low risk under the real deployment distribution) and what a single Random Forest on tabular photometry achieves, then narrows to actionable work.

---

## 1. Approximation error: is the hypothesis class too small?

The Bayes-optimal decision rule may be **complex** (non-axis-aligned, interactions at many scales). A finite-depth forest with limited trees only approximates that rule. **Bias** in the learning-theory sense remains even with infinite data.

**Concrete direction:** Increase **capacity** in a controlled way—deeper trees, more trees, or a **richer class** (gradient boosting with shallow base learners often approximates smoother boundaries than a single RF of comparable size). **Search** hyperparameters (depth, `min_samples_leaf`, learning rate if boosting) on a validation fold so gains are not luck.

---

## 2. Estimation error: do we need more or better data?

With finite \(n\), we estimate \(f\) with **variance**. Collecting more **i.i.d.** data from the same \(P(x,y)\) tightens generalization bounds in expectation; **active** selection helps if labelling spectra is expensive.

**Concrete direction:** For this repo, \(n = 10^5\) is already large for tabular RF; the bottleneck is often **distribution shift** (below), not raw count. If subsampling for speed, stratify by class and by redshift.

---

## 3. Prior shift and cost-sensitive decisions

The **base rate** \(P(y)\) in the catalogue may differ from the sky population you care about. The Bayes rule depends on **priors**; a model tuned for one \(P(y)\) is **not optimal** when priors change unless you **reweight** or adjust thresholds.

**Concrete direction:** Use `class_weight`, explicit loss matrices, or resampling so **precision/recall per class** match science goals (e.g. rare quasar completeness vs purity).

---

## 4. Calibration: when probabilities must mean frequencies

If downstream decisions use a **cut on \(P(y|x)\)** (e.g. trigger spectroscopic follow-up when \(P(\text{QSO}) > 0.9\)), you need **calibrated** probabilities: predicted 0.9 should occur on ~90% of such predictions over the long run **on the deployment distribution**.

RF and boosted trees are often **miscalibrated** out of the box.

**Concrete direction:** Post-hoc calibration on a **held-out** set—Platt scaling, isotonic regression, or temperature scaling on logits if you add a parametric head.

---

## 5. Representation: sufficiency of features

No algorithm can recover information **not present** in \(x\). If \(x\) is **photometry only**, then any two classes with **overlapping** class-conditional densities \(p(x \mid y)\) incur **irreducible error** regardless of model.

**Concrete direction:** Add **minimal sufficient statistics** astronomers use: explicit **colour indices**, extendedness/morphology, variability, or low-resolution **spectral features**. Each addition moves \(x\) toward a space where **overlap shrinks** (if the new observable carries new information).

---

## 6. Generalization beyond this catalogue

Risk on a random test row from the **same CSV** estimates **in-sample** generalization only. **Domain shift**—new observing system, depth, dust maps, selection—changes \(P(x)\) or \(P(y \mid x)\).

**Concrete direction:** **External validation**: train on one survey slice, plate range, or magnitude bin; test on another. Report **degradation**, not just headline accuracy.

---

## 7. Uncertainty beyond a point estimate

A single label \(\hat{y}\) throws away information. **Epistemic** uncertainty (model unsure) differs from **aleatoric** uncertainty (overlap in \(p(x|y)\)).

**Concrete direction:** Ensembles, dropout-free **deep** models with MC dropout (if used), **conformal prediction** for finite-sample coverage guarantees, or **evidential** heads—choose based on whether you need **intervals** or **class sets**.

---

## 8. Engineering: reproducible experiments

Science requires that **claims** (tables, figures) be **functions** of declared code, data version, and seed.

**Concrete direction:** Config files for paths and hyperparameters; CI smoke tests on tiny extracts; notebook outputs treated as **build artifacts** when publishing.

---

*See `01-lessons-learned.md` for how the current baseline fits into this picture.*
