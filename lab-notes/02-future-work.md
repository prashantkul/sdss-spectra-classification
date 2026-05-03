# 02 — Future work

Concrete next steps if this project grows beyond the Random Forest baseline.

## Modelling

1. **Hyperparameter search** — `GridSearchCV` / `RandomizedSearchCV` on `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features`. The current settings are a reasonable default, not an optimum.

2. **Stronger tabular learners** — Gradient-boosted trees (XGBoost, LightGBM, CatBoost) often gain a few points on mixed feature types and can handle mild imbalance with `scale_pos_weight` / class weights.

3. **Class weights** — If the training distribution drifts from the sky (e.g. oversampling quasars), retrain with `class_weight='balanced'` or explicit weights and re-check per-class precision/recall.

4. **Calibration** — Temperature scaling or isotonic regression on validation data so reported probabilities match empirical frequencies—useful if downstream decisions use a probability threshold.

5. **Ensembles** — Blend RF + GBM with stacked logits or simple averaging on difficult pairs (especially QSO vs GALAXY).

## Features

6. **Colour indices** — Engineered features such as `u−g`, `g−r`, `r−i`, `i−z` reduce redundancy and align with how astronomers separate loci in colour–colour space.

7. **Morphology / extendedness** — If available (e.g. from imaging), separating point-like vs resolved sources directly attacks the QSO–galaxy confusion seen in photometry-only models.

8. **Spectra** — The endgame for ambiguous cases: even low-S/N classification from line templates outperforms photometry where manifolds overlap.

## Evaluation

9. **Stratified reporting by redshift and magnitude bins** — Performance often degrades at faint limits or high z; slice the test set accordingly.

10. **Cross-survey generalization** — Train on one footprint or release and test on another to measure catalogue bias, not just IID test accuracy.

11. **Uncertainty** — Conformal prediction or ensemble disagreement for error bars on each class probability.

## Engineering

12. **CLI + config** — YAML or Hydra for paths, seeds, and model YAML so `main.py` runs match notebook experiments.

13. **Tests** — Smoke tests on a tiny CSV subset to guard `load_data` and encoding logic in CI.
