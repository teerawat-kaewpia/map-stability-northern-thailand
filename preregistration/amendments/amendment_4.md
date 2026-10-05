<!-- extracted from predeclaration_2026-10-01_redacted.md -->
### Amendment 4 - 2026-10-01, before any background is drawn or any model is fitted

**Why.** Writing the runner exposed procedures that the original text and Amendments 1-3 do not fix,
and one error in the lock. Each item is implemented in `01_scripts/run_map_stability.py` (SHA-256 in
the lock under `"runner"`) under the matching label `U1`-`U14`. The runner refuses `--run` unless the lock
records this amendment and that hash. No background had been drawn and no model fitted on study data
when this was written; the runner had only been exercised in a dry run and on synthetic data.

**A4.0 Border proximity (adds to A3.2).** Training background can lie close to a test presence along any
provincial boundary, not only at the six registry-polygon mismatches; those six are examples found,
not the full set.

**A4.1 Correction to the lock.** The lock described `paperA_validation_utils.py` as defining "the fixed
50 km spatial-block grid". That was wrong: its `spatial_block_labels` anchors the grid at the minimum
easting/northing of the data passed in, so the grid would move with every background draw, which A1.5
excludes. The runner does not use that file; its lock entry is kept for traceability only.

**A4.2 Spatial blocks (U2).** Inner 50 km blocks are an absolute grid, `(floor(E / 50000),
floor(N / 50000))`, as in Paper C's leave-one-province-out `spatial_block_auc`.

**A4.3 Threshold (U3).** Each learner uses its own leave-one-block-out out-of-fold scores on its fold's
training data (training presences + that unit's training background). A block is skipped if the rest of
the training data lacks a class. A TSS tie is any value within `1e-12` of the maximum, because values
that are mathematically equal can differ in the last floating-point digits; among ties the highest
finite threshold is taken (A1.6).

**A4.4 Map grid and mask (U1).** 500 m grid from the locked boundary (union of the nine polygons), origin
floored/ceiled to 500 m; mask = rasterised boundary AND all 11 covariates finite; covariates warped from
the locked rasters, bilinear except forest and built-up (nearest); the five distance rasters converted
from m to km; `dist_border_CK_km` computed at cell centres (A1.8). Dry run: 847 x 737 cells, 382,056
valid. Paper C's published map grid used GADM 4.1 (382,067 valid cells), so maps from this experiment
are compared only with each other.

**A4.5 Map models (U4).** Fitted on all nine provinces (`n_bg = 2,859`, A3.1) for every radius x learner x
seed: 90 maps. Similarity is computed within a seed, then summarised for each comparison, metric and
province as the median, minimum and maximum across the five seeds, with the number of seeds that had a
value. A map whose background is incomplete under the budget is reported as missing in every comparison
that needs it.

**A4.6 SSIM and NoData (U5).** `skimage.metrics.structural_similarity(full=True, win_size=w,
data_range=1)` with its default uniform window. NoData is filled with 0 only to compute the SSIM map,
which is then averaged over a province's cells whose **entire w x w window is valid**; windows touching
NoData or the array edge are excluded rather than scored.

**A4.7 Percentile-rank maps (U6).** Rank over all valid study-area cells, `rankdata(method="average") / n`.

**A4.8 Top 10% (U7).** Cells with score >= the 90th percentile of all valid study-area cells of that map
(one study-wide threshold, matching the regional screening question). Per province, Jaccard is reported
**together with the counts behind it** - top-10% cells in each map, intersection, union - so a NaN from
an empty union, or a Jaccard resting on very few cells, is visible. For seed 42 a gain/loss/stable class
map is saved for every compared pair.

**A4.9 Permutation importance (U8).** On each fold's shared test set; drop in AUC; 10 repeats; a fresh
`default_rng(42)` per factor or group; built-up and nightlight permuted with one shared row permutation;
every fold, radius, learner and seed.

**A4.10 SHAP (U9).** Seed-42 map models only (18). Explained: 1,000 valid cells drawn with
`default_rng(314159)`. Reference background: 100 valid cells drawn with `default_rng(271828)`. Output is
P(presence) for all three learners. Random forest and XGBoost: `TreeExplainer(..., feature_perturbation=
"interventional", model_output="probability")`. Logistic regression: shap's **PermutationExplainer** on
`predict_proba[:, 1]` with an Independent masker on the same 100 cells, `seed = 161803`, `max_evals = 115`
(five permutations of 2 x 11 + 1). The logistic-regression values are therefore a sampled estimate, while
the tree values are exact for the interventional definition; all three are on the probability scale with
the same reference cells. Direction = Spearman correlation between a factor's value and its SHAP value.
The summary is rebuilt from every valid SHAP file on disk, with a coverage record of which of the 18
models are present.

**A4.11 Leave-one-factor-out (U10).** Seed 42 only, every fold x radius x learner x the five pre-declared
groups; refit without the group; AUC drop on the fold's shared test set; no threshold. A supplementary
result: its conclusions are conditional on the seed-42 background draw.

**A4.12 Metrics (U11).** TSS = sensitivity + specificity - 1 and Cohen's kappa, both at the frozen
threshold, positive if score >= t. For each learner x radius x seed the spread across the nine folds is
reported as minimum, lower quartile, median, upper quartile and maximum; fold medians are then summarised
across seeds as median, minimum and maximum.

**A4.13 No imputation (U12, U13).** Every training row, test row and mapped cell must have finite
covariates; the runner asserts this instead of imputing. Presence covariates are used as stored in the
locked clean presence file.

**A4.14 Interruption and resume (U14).** Every output is written to a temporary file and renamed only when
complete. On resume a unit is skipped only if its files load and pass content checks: a leave-one-province-
out unit's JSON (written last, as the commit marker) must parse and contain all three learners with every
required field, and its background file must have the expected shape; a map must have one finite value
in [0, 1] per valid cell; seed-42 models must load; a SHAP file must be 1,000 x 11.

**Run time is not estimated here.** No timing of the real workload has been made.

**Not changed by this amendment:** the design, the test points, the sampling rules of Amendments 1-3,
learner settings and the reading rule.
