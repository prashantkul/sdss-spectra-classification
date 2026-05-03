# 01 — Lessons learned

Notes from building the SDSS-style stellar classification baseline (`sdss_analysis.ipynb`, `main.py`, `analysis.py`).

1. **Photometry alone is surprisingly strong.** Magnitudes `u, g, r, i, z` plus redshift and coordinates push test accuracy into the high-90s for this Kaggle slice. That says more about how separable the catalogue is than about “solving” astronomy: spectroscopic labels were used to build the training set, so the model is learning the same observables experts already used for pre-selection.

2. **Scale before distance-based models (and forests still benefit).** `StandardScaler` on continuous inputs keeps features comparable. Even tree ensembles, which don’t *require* scaling, sit in pipelines where scaling is harmless and keeps experiments consistent if you swap in k-NN, logistic regression, or neural nets later.

3. **Leakage is easy to miss.** IDs (`obj_ID`, `spec_obj_ID`, run/plate/fiber) must stay out of features unless you intentionally want to memorize survey artefacts. Dropping or holding out correlated proxies is part of “tabular hygiene.”

4. **Feature importance is directional, not causal.** Random Forest `feature_importances_` shows what the forest split on most often, not physical causation. Correlated features (colours, redshift) share credit; interpret with correlations and partial dependence if you need stronger claims.

5. **One confused pair dominates the error budget.** On the test run in the notebook, **QSO → GALAXY** accounted for most misclassifications. Diagnose errors by **(true → predicted)** pairs, not only overall accuracy—otherwise you optimize the wrong thing.

6. **High `predict_proba` on a wrong class is a feature, not a bug.** It flags **boundary** objects where the forest is overconfident. Those rows are the right ones to inspect manually or send to a second-stage model.

7. **Reproducibility needs pinned environments.** Notebook outputs (figures in `docs/plots/`) reflect one sklearn / numpy / seed combination. Lock files (`uv.lock`) and `requirements.txt` avoid “works on my machine” drift.

8. **External data deserves explicit citation.** The CSV traces to a specific Kaggle dataset (see repo README). Lab notebooks should always record *where* data came from and *what* the label definition is (spectroscopic class, survey release, etc.).
