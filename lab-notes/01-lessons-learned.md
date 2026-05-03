# 01 — Lessons learned (from first principles)

What follow are not “tips,” but implications of a few basic definitions, applied to this repo’s photometric classification task.

---

## 1. What problem we are actually solving

**Supervised classification:** we observe a feature vector \(x\) (magnitudes, positions, redshift, etc.) and want a mapping \(f(x)\) that predicts a discrete label \(y \in \{\text{STAR}, \text{GALAXY}, \text{QSO}\}\).

Training chooses \(f\) from a **hypothesis class** (here: piecewise-constant surfaces induced by a Random Forest on scaled \(x\)) by minimizing **empirical risk** on a finite sample \(\{(x_i, y_i)\}\). At deployment, we care about **risk on new draws** from the same (or a shifted) joint distribution \(P(x, y)\).

**Lesson:** High accuracy on a held-out split means the empirical distribution of this catalogue is **partially predictable** from \(x\) with the learner we chose—nothing more. It does not prove we have “understood” astrophysics; it means the survey’s label-assignment process left **recoverable structure** in the published features.

---

## 2. Why good photometry is enough (sometimes)

The **Bayes classifier** assigns \(x\) to the class with largest **posterior** \(P(y \mid x) \propto P(x \mid y)\,P(y)\). If, for most \(x\) in the catalogue, one class dominates the posterior, then **any** reasonable learner will score well.

Spectroscopic targets are pre-selected using photometry and other cuts; labels themselves came from spectra applied to objects that already occupy a **structured corner** of feature space. The model can therefore reach the high-90% range without spectra: it is learning the same **low-dimensional summaries** of \(x\) that the survey encoded in the table.

**Lesson:** Strength here reflects **label/feature alignment** in this dataset, not a guarantee that photometry alone resolves every object in nature.

---

## 3. Scaling: invariance and comparability

**StandardScaler** applies an affine transform \(\tilde{x} = (x - \mu)/\sigma\) per feature (fit on training data only). Tree ensembles are largely **invariant** to monotone transforms of individual features, so scaling does not change the **optimal** axis-aligned splits in principle.

**First-principle reason to scale anyway:** (1) training pipelines stay **comparable** when you swap in distance- or margin-based models (same preprocessing story); (2) numerical stability; (3) **experimental hygiene**—features stay on comparable numeric scales in notebooks and tables.

---

## 4. Leakage: violating the deployment conditional

A feature is valid if, at prediction time, it would be **known without peeking at the answer we are trying to predict**. Survey IDs (plate, MJDs, `obj_ID`) can correlate with **instrument, epoch, or targeting** and indirectly with class. Including them lets the model exploit **spurious training-set structure** that need not hold under new survey conditions.

**Lesson:** Exclude variables that are not part of the **causal or operational** input you intend to use in production, even if they improve offline accuracy.

---

## 5. Feature importance ≠ physics

Random Forest importance (mean decrease in impurity, or permutation importance) measures **contribution to predictive splits** under correlation with other features. It answers: “If we disrupt this coordinate, how much does the fitted forest’s loss change?” not “What caused the object to be a quasar?”

Correlated inputs **share** importance; reordering or merging redundant magnitudes changes the allocation of importance without changing predictions much.

**Lesson:** Treat importance as a **debugging and ranking** tool for the fitted model, not as a theory of emission mechanisms.

---

## 6. Evaluation: accuracy is a summary, not the full story

Accuracy is \(P(\hat{Y} = Y)\) under the test **empirical** distribution. A single scalar **aggregates** a **confusion matrix**—the joint frequencies of \((Y, \hat{Y})\). Different confusion patterns can yield the same accuracy.

**Lesson:** Inspect **per-class precision/recall** and dominant **(true → predicted)** pairs. In this project, one pair (**QSO → GALAXY**) dominates the error budget; optimizing only global accuracy can hide a **biased** failure mode.

---

## 7. Probabilities from forests: calibrated?

`predict_proba` for Random Forests is **vote / leaf frequency** among trees. It is not guaranteed to match **long-run event frequencies** (it can be overconfident).

**Lesson:** High `predict_proba` on a **wrong** class means the ensemble is **internally unanimous** near a sharp partition—not that the object “is almost certainly” that class in a frequentist sense. Such rows are ideal **hard cases** for review or a second model.

---

## 8. Reproducibility = same program + same inputs → same outputs

Finite differences in library versions, BLAS threads, or data ordering can change splits and tie-breaking in tree learners. **Pinned dependencies** and fixed **random seeds** (where the algorithm respects them) turn “experiment” into a **repeatable transformation**.

**Lesson:** Commit lockfiles; treat figures in `docs/plots/` as artifacts of one **recorded** environment.

---

## 9. Provenance is part of the model

The mapping \(f\) is fit to **rows from a particular release and labelling policy** (Kaggle SDSS17 stellar classification dataset). Changing pipelines or spectroscopic definitions changes \(P(y \mid x)\).

**Lesson:** Cite the dataset and the label definition whenever you report metrics; otherwise results are not **comparable** across papers or repos.

---

*For runnable numbers and plots, see `sdss_analysis.ipynb` and `GETTING_STARTED.md`.*
