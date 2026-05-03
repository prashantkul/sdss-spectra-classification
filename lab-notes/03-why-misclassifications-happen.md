# 03 — Why misclassifications happen

This note summarizes how errors arise in **this** project’s photometry-only classifier. It mirrors section **11c** of `sdss_analysis.ipynb`; numbers (e.g. error rate, dominant pair) can change slightly if you retrain or change seeds—rerun the notebook for the live crosstabs.

## 1. Dominant confusion: QSO ↔ galaxy (especially QSO predicted as GALAXY)

Quasars are unresolved AGN; in single-epoch SDSS-like photometry they can look like compact galaxies or red point sources. When the true label is **QSO** but the model predicts **GALAXY**, photometric colours and redshift can sit in the same region occupied by real galaxies. That single direction often accounts for the largest share of test errors.

## 2. Overlap in colour–redshift space

Stars, galaxies, and quasars separate *on average* in diagrams such as **u−g versus redshift**, but the loci intersect. Misclassified points frequently **pile up in those wedges** where class manifolds cross. The model is doing interpolation in the feature space; near intersections, small noise or measurement error flips the majority vote.

## 3. Compact and star-forming galaxies

**Galaxies** with compact profiles or strong star formation can mimic **point-like** sources in coarse photometry, moving them toward the quasar/star locus. Dust **extinction** and **reddening** further smear colours, widening overlap between classes.

## 4. Label noise and “impossible” perfection

Training labels come from **spectroscopic classification** in a survey pipeline. That process has its own failures: bad fits, composite spectra, mis-z, or objects that are genuinely ambiguous. The notebook reports on the order of **~2–3%** of test rows disagreeing with the label the forest was trained on; some fraction may be **catalogue inconsistency**, not model stupidity.

## 5. High confidence on wrong predictions

Random Forests can assign **high `predict_proba` max** on errors. That usually means the tree ensemble is **sharp at a decision boundary** that does not match the human/spectroscopic class for that object. These rows are high value for review: either the model is wrong (needs features or capacity) or the label is edge-case.

## 6. What this note is *not*

- **Not causal physics** — We describe statistical overlap in feature space, not proven ionization or host-galaxy mechanisms.
- **Not complete** — Morphology, variability, and spectra are absent here; many failure modes would shrink with richer inputs.

For plots (confusion pairs, colour–redshift scatter of errors), see the notebook section **“11c. Misclassification analysis”** and `docs/plots/misclassification-analysis.png` in the repo.
