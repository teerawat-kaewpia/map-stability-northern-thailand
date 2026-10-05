# Map-stability experiment - pre-declaration

Written **2026-10-01, before the experiment was run**. Nothing below was chosen after seeing a
result. Any later deviation is recorded in this file as a numbered amendment with its reason, and is
never edited away.

## The question

When the background sampling radius changes, or the learner changes, **does the land that the map
calls suitable stay in the same place** - and if it moves, what moves it?

The question is deliberately not "which configuration scores highest". Two maps can discriminate
equally well and still send a planner to different places, and that disagreement is the object of
study here.

## Separation from Paper C

This is a separate experiment with its own lock. It does not reference, extend or supersede
`paper_C/03_results/preregistration/artifact_lock_2026-09-27.json`. Paper C's transfer results,
Figure 9 and its four per-fold tables are **read-only** here: they are not recomputed, not
regenerated and not overwritten. Outputs land only under `RF_RP/exp_map_stability/`.

## Design

| Factor | Levels |
|---|---|
| Background ring, outer radius | 1 km, 5 km, 10 km, 20 km, 25 km, and **no outer bound** (written `R-inf`) |
| Learner | random forest, XGBoost, logistic regression |
| Covariates | the same eleven in every configuration |
| Configurations | **18** = 6 radii x 3 learners |
| Seeds | **5 pre-declared seeds**: 42, 0, 1, 7, 2024 |

`R-inf` means **no outer bound on the ring**, with sampling still confined to the nine-province
polygon. It is not infinite extent.

"1 km ring" is the name used throughout this file, so that it cannot be confused with **rule R1** of
Paper C's protocol.

## Fixed before the run

1. The same presence set, the same covariate grids and the same nine-province boundary in every
   configuration. The inner exclusion stays at **500 m** throughout.
2. Background draws are the **same size** for every radius, taken from the **same candidate
   ordering**, under each of the five seeds. Five seeds exist so that the effect of the radius can be
   separated from the accident of a particular draw.
3. **Infeasibility rule.** If the 1 km ring cannot yield the required number of usable background
   points under these criteria, the result is reported as *not computable under the locked criteria*.
   The ring is not widened, the sample size is not reduced, and no rule is changed after the fact.
4. Learner settings are fixed now and not tuned per configuration:
   - random forest: 500 trees, `max_features` the square root of the predictor count, minimum leaf 5,
     balanced class weights
   - XGBoost: 500 rounds, depth 4, learning rate 0.05, subsample 0.8, colsample 0.8, logloss
   - logistic regression: standardised predictors, L2, C = 1.0, lbfgs, max_iter 2000, balanced
5. Every map is produced on the **same 500 m grid, the same extent and the same valid-cell mask**.
   **No per-map min-max rescaling.** Rescaling each surface to its own range would erase exactly the
   difference in score level that this experiment is measuring.
6. Evaluation is **leave-one-province-out, nine folds**. In each fold, every configuration is scored
   against the **same held-out presence and background test points** for that province. Test points
   are never used for fitting or for choosing a threshold.

## What will be reported

### 1. Classification - a supporting result, not the headline

AUC, TSS and Kappa **per province**, with the median and the spread across folds. Thresholds are
chosen on training data only and then applied unchanged to the held-out province.

TSS and Kappa are labelled as performance against **sampled background**, not as specificity against
confirmed absence. Comparing AUC across radii while the test background differs between them would
answer a different question from the one asked, so the test background is shared
(Barbet-Massin et al., 2012).

### 2. Map similarity - the headline result

Two directions: **across radii within one learner**, taking the published 10 km ring as the
reference, and **across learners within one radius**.

- **SSIM on the raw scores** and **SSIM on the percentile-rank maps**, reported separately. The first
  responds to both the score level and the spatial pattern; the second isolates whether the ordering
  of places changed.
- **Jaccard overlap of the top 10% of scores**, with difference maps, to show where the screening
  area moves to.
- Values reported **per province**. Pixels are not treated as independent samples, and no
  significance test is run over pixel counts.

SSIM window is fixed in advance: **7 x 7 cells as the primary scale**, with one other window as a
sensitivity check. `data_range` is 1 for scores on 0-1, and NoData is excluded rather than counted as
zero (scikit-image documentation).

### 3. Factor use - the explanatory result

**Permutation importance measured as the drop in AUC on the shared test set** is the cross-learner
comparison, because it is on one scale for all three learners.

**SHAP** is then used on the **same sample of cells** to check rank, direction and where a factor
acts. Before any cross-learner comparison, SHAP output scale and reference background are made to
match: XGBoost's default explanations can be in log-odds, so raw magnitudes are not comparable with
the random forest's without that step (SHAP documentation).

Correlated factors, **built-up and nightlight in particular, are also tested as a group**, because
permutation importance can be low for each of a correlated pair while the pair matters
(scikit-learn documentation).

**Leave-one-factor-out** is a supplementary test, run only for factor groups named in advance,
because each run requires refitting.

## The rule for reading the result

**If AUC values are close but SSIM or hotspot overlap is low, the finding is that the configurations
discriminate comparably while giving different spatial advice.** That is the headline of this
experiment. No map is selected on the strength of the highest AUC alone.

## Pre-declared factor groups for the supplementary test

1. development intensity: built-up, nightlight
2. transport access: distance to road, distance to existing railway, distance to the planned alignment
3. market and gateway access: distance to CBD, distance to dry port, distance to the Chiang Khong crossing
4. physical capacity: slope, distance to water
5. forest-area condition: forest

## Inputs, locked before the run

`input_lock_2026-10-01.json` records the SHA-256 of all **13** inputs: the presence file, the
analysis dataset, the province boundary and the ten covariate rasters.

It also records the **provenance of the nightlight raster**, as required before running: the layer is
the **Version 2** NPP-VIIRS-like annual product for 2024, distributed under CC0 1.0 through Harvard
Dataverse, and produced for the years used here by a **deep-learning super-resolution
reconstruction** (Chen et al., 2026). The 2021 ESSD paper documents the earlier
cross-sensor-calibrated series and covers 2000-2018 only, so it does not document this layer. The
grid was derived on 2026-05-08, before the provider re-released 2022-2024 in July 2026; the earlier
release is kept deliberately, so that this experiment uses the same covariate values as the locked
Paper C results. It is not a raw VIIRS observation and carries no legal or administrative attribute.

## Amendments

None yet.

### Amendment 1 - 2026-10-01, before any run

**Why.** A review of this file against the input files, made before any runner existed and before
any result, found (a) the locked presence file has the wrong role, (b) the nightlight provenance
states a version and method that the evidence contradicts, and (c) four procedures were not fixed
tightly enough to rule out choices made after seeing results. No result exists. The text above is
left unchanged; where it conflicts with this amendment, this amendment governs.

**A1.1 Training presence input.** The presence set used for fitting is
`RF_RP/data/processed/analysis_dataset_m3_base_clean.csv`: 2,859 rows, all `presence = 1`
(SHA-256 `699d63aff5203528d74f1d4fc4f5d9df13347a97c8ac3da27a3c4b8d5b876665`). The previously locked
`paper_C/03_results/data/presence_base_9prov_full.csv` has 2,868 rows with `presence = 1` on only
734 (the pre-QC, field-confirmed-only labelling); it stays in the lock as source evidence and **must
not be read by the runner**. `analysis_dataset_m3_full_r10.csv` is reference only: its 2,859
presences have the same FIDs as the clean file (checked), but backgrounds are regenerated per radius.

**A1.2 Province membership.** A presence belongs to the province in its `FPROVNAME` field, as in
Paper C's leave-one-province-out runner. Province polygons are the nine features of the locked
boundary file (`ADM1_TH`, codes TH50-TH58). Six presences lie inside a different province polygon
from their `FPROVNAME`; they are kept under `FPROVNAME` and not reassigned. Presences per province:
Chiang Mai 906, Chiang Rai 559, Lampang 468, Lamphun 332, Uttaradit 201, Nan 146, Phrae 113,
Phayao 81, Mae Hong Son 53 (total 2,859).

**A1.3 Training background - confined to training provinces, every radius.** In each fold, training
background for every radius, `R-inf` included, is drawn only inside the union of the eight training
provinces' polygons, around training presences only. The held-out province contributes nothing to
fitting, to the choice of threshold, or to any inner validation. (This generalises the instruction
given for `R-inf`; it is also how Paper C's leave-one-province-out runner treats training data.)

**A1.4 Test background - shared, generated once.** For each held-out province: points drawn
uniformly inside that province's polygon, at least 500 m from **all 2,859 known presences**, in a
number equal to that province's presence count (A1.2). Fixed seed **314159**, separate from the five
experimental seeds. Candidates are drawn uniformly in the province's bounding box and kept in draw
order if inside the polygon, at least 500 m from every presence, and with no missing covariate;
the first *n* kept are used. If fewer than *n* qualify, the fold is reported as *not computable
under the locked criteria* (same rule as item 3 above). The nine test sets are generated **once**, in
a separate step before any model is fitted, written to `exp_map_stability/02_data/test_points/`, and
their SHA-256 recorded in the input lock before the first fit. Every radius, learner and seed is
scored on these same points.

**A1.5 Seeds.** The five seeds (42, 0, 1, 7, 2024) change **only the ordering of the training
background candidates**. Random forest and XGBoost use `random_state = 42` in every run; logistic
regression is deterministic. Inner spatial blocks are the fixed 50 km grid and do not depend on the
seed. Variation across the five seeds is therefore attributable to the background draw alone.

**A1.6 Threshold.** In each fold the threshold is chosen from out-of-fold predictions of a 50 km
spatial-block cross-validation run **inside the training provinces only**. Candidate thresholds are
the distinct finite out-of-fold scores; the one maximising TSS is chosen. Ties are broken by taking
the **highest finite** threshold (`+inf` is never a candidate). A cell or point is predicted positive
when its score is **>= threshold**. The threshold is then frozen and applied unchanged to compute TSS
and Kappa on the held-out province.

**A1.7 SSIM sensitivity window.** **21 x 21 cells** (about 10.5 km on the 500 m grid), reported beside
the primary 7 x 7 window (about 3.5 km). `data_range = 1` and the NoData rule are unchanged.

**A1.8 Eleventh covariate, `dist_border_CK_km`.** Not a locked raster; computed from coordinates.
Chiang Khong customs at **20.3622 N, 100.0894 E** (WGS 84), transformed to EPSG:32647 with
`pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)`; distance in km =
`sqrt((E - E_ck)^2 + (N - N_ck)^2) / 1000`. Implemented for points in
`paper_C/03_results/scripts/phase1b_radius_comparison.py` and for the 500 m prediction grid in
`paper_C/03_results/scripts/paperA_suitability_uncertainty_maps.py` (which overrides the
`D15_dist_border_CK_km_500m.tif` raster it first loads). Both scripts are hashed in the input lock.
Euclidean, not network distance; that limitation is unchanged.

**A1.9 Nightlight provenance - claims withdrawn.** The statements in "Inputs, locked before the run"
that the layer is the **Version 2** product and was produced by **deep-learning super-resolution
reconstruction** (Chen et al., 2026) are **withdrawn**: the evidence below contradicts them. What
can be verified: the model grid `grid_nl_17prov.tif` (hash in the lock, modified 2026-05-08) was
derived from `LongNTL_2024.tif` in a provider folder named `2024_HasMask_Version1`; that file's ArcGIS
sidecar records it as a corrected VIIRS 2024 median composite multiplied by a global mask, computed
2025-04-12 in the producer's working directories (`ChenZuoqi\LongNTL`). Producer: Zuoqi Chen and
co-authors, distributed through Harvard Dataverse doi:10.7910/DVN/YGIVCD. **Version and release:
undetermined.** The file predates both the provider's December 2025 update of the Version 2 1992-2024
data (with the notice that Version 2 data downloaded before December 2025 must be re-downloaded) and
the July 2026 re-release of 2022-2024. **The raster values are not changed**, so this experiment uses
the same covariate values as the locked Paper C results.

**Not changed by this amendment:** factors, levels, the 18 configurations, learner settings, the
map grid and mask, LOPO with nine folds, the map-similarity measures and the reading rule.

### Amendment 2 - 2026-10-01, before any run and before the test points exist

**Why.** A review of Amendment 1 (no run, no result, no test points generated) found that one sentence
in A1.3 overstates the separation achieved by a registry-based split, and that the infeasibility
rule in A1.4 has no stopping point. Amendment 1 is left unchanged; where they conflict, this
amendment governs.

**A2.1 What the leave-one-province-out split does and does not separate (corrects A1.3).** The
sentence in A1.3 "The held-out province contributes nothing to fitting" is **withdrawn as worded**.
Folds are defined by registry province (`FPROVNAME`, A1.2), so what holds is: **no presence whose
registry province is the held-out province is in the training set.** It is **not** a strictly
geographic holdout. A spatial join of the 2,859 clean presences with the locked province polygons
(run 2026-10-01) finds six presences whose registry province differs from the polygon they lie in:

| FID | registry province | lies in polygon of | distance to that polygon's edge |
|---|---|---|---|
| [registry ID 1 redacted] | Lamphun | Chiang Mai | 19,823 m |
| [registry ID 2 redacted] | Lamphun | Chiang Mai | 17,275 m |
| [registry ID 3 redacted] | Chiang Mai | Lamphun | 926 m |
| [registry ID 4 redacted] | Chiang Mai | Lamphun | 1,958 m |
| [registry ID 5 redacted] | Chiang Rai | Phayao | 75 m |
| [registry ID 6 redacted] | Phrae | Nan | 5,345 m |

This creates exceptions in **six folds, in two directions**:
- *training presence located inside the held-out polygon*: Chiang Mai fold (2), Lamphun fold (2),
  Nan fold (1), Phayao fold (1);
- *test presence located outside the held-out polygon, while that fold's test background is drawn
  inside it (A1.4)*: Chiang Mai fold (2), Lamphun fold (2), Phrae fold (1), Chiang Rai fold (1).

These records are kept as they are: no presence is reassigned or removed, so that the folds match
Paper C. The six FIDs and the affected folds are reported with the per-fold results. Training
**background** remains confined to the training provinces' polygons at every radius, `R-inf`
included (A1.3 unchanged on this point; it is also how Paper C's runner passes the training-province
polygon to the sampler, `paperC_spatial_transfer_lopo.py` line 276).

**A2.2 Sampling budget for the test points (completes A1.4).** For each held-out province at most
**max(20,000, 200 x n_test)** candidates are drawn, where a candidate is one uniform draw in the
province's bounding box, counted before any filter. The first `n_test` candidates that pass all
A1.4 filters, in draw order, are used. If fewer than `n_test` pass within the budget, that fold is
reported as **"incomplete under the locked sampling budget"** - not as evidence that the province has
no usable points - and the budget is not raised after the fact. Budgets: Chiang Mai 181,200;
Chiang Rai 111,800; Lampang 93,600; Lamphun 66,400; Uttaradit 40,200; Nan 29,200; Phrae 22,600;
Phayao 20,000; Mae Hong Son 20,000.

**Not changed by this amendment:** everything in Amendment 1 other than the withdrawn sentence of
A1.3 and the stopping rule of A1.4.

### Amendment 3 - 2026-10-01, before any run and before the test points exist

**Why.** Item 3 above ("if the 1 km ring cannot yield the required number ... not computable") had no
stopping point for the training-background draw, and the number of training background points in a
leave-one-province-out fold was not fixed. Without both, whether a configuration is "computable" could
be decided after seeing it. Amendments 1-2 are left unchanged.

**A3.1 Training background size.** In each fold `n_bg = n_presence_train = 2,859 - n_presence_test`
(1:1, as in Paper C). Models fitted on all nine provinces to produce the maps use `n_bg = 2,859`.

| held-out province | test presences | `n_bg` (training) | round-1 draw | early-stop at |
|---|---|---|---|---|
| Chiang Mai | 906 | 1,953 | 39,060 | 5,859 |
| Chiang Rai | 559 | 2,300 | 46,000 | 6,900 |
| Lampang | 468 | 2,391 | 47,820 | 7,173 |
| Lamphun | 332 | 2,527 | 50,540 | 7,581 |
| Uttaradit | 201 | 2,658 | 53,160 | 7,974 |
| Nan | 146 | 2,713 | 54,260 | 8,139 |
| Phrae | 113 | 2,746 | 54,920 | 8,238 |
| Phayao | 81 | 2,778 | 55,560 | 8,334 |
| Mae Hong Son | 53 | 2,806 | 56,120 | 8,418 |
| all nine (map models) | - | 2,859 | 57,180 | 8,577 |

**A3.2 Training-sampling budget - Paper C's rule, every radius and every seed.** The rule is the one
implemented in `paper_C/03_results/scripts/paperC_spatial_transfer_lopo.py`, lines 121-148 (file
hashed in the input lock):
- One generator `numpy.random.default_rng(seed)` is created at the start of each draw and continues
  across rounds.
- At most **6 rounds**. Round *i* (1-6) draws `k_i = i x max(20 x n_bg, 20,000)` points uniformly in the
  bounding box of the sampling polygon - eastings first, then northings, as in that code. The sampling
  polygon is the union of the training provinces in a fold, or the nine-province polygon for map
  models.
- A point is kept if it lies inside the sampling polygon, at least 500 m from the nearest **training**
  presence, and within the outer radius from it. Kept points accumulate in draw order. Drawing stops
  after round *i* once at least `3 x n_bg` points have been kept.
- Covariates are then extracted for all kept points; points with any missing covariate are dropped;
  the first `n_bg` remaining, in draw order, are used.
- If fewer than `n_bg` remain, that fold / configuration / seed is recorded as **"incomplete under the
  locked training-sampling budget"**. No round is added and `n_bg` is not changed.

Distances for the ring and the 500 m exclusion are measured to **training presences only**, as in
Paper C. A training background point can therefore lie within 500 m of a test presence that sits
outside its own province polygon (the exceptions listed in A2.1).

**A3.3 Paired comparison.** Within a fold and a seed, the draw for every radius starts a fresh
generator from that same seed. Because `n_bg` - and therefore every `k_i` - is the same for all radii in
a fold, all radii see the identical candidate sequence; they differ only in the ring filter and, as a
consequence, possibly in how many rounds are drawn before the stop. The background drawn for a fold,
radius and seed is generated once and **shared by all three learners**, so learner comparisons within
a radius are paired as well.

**A3.4 `R-inf`.** Same rule with only the outer-radius condition removed (the inner 500 m exclusion and
the polygon remain). Paper C's function takes the outer radius as a number, so this is implemented in
the new runner; the old function is not called with `None`.

**A3.5 What the earlier evidence does and does not show.** In Objective_One a 1 km ring yielded 822 of
2,868 points under a single draw of 20 x n. That motivated locking a budget; it is **not** evidence
that the 1 km ring will be incomplete under the six-round rule above, and no expectation about that
outcome is recorded here.

**Not changed by this amendment:** everything in Amendments 1-2 and in the original text other than
the stopping rule and sample size of the training background.

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
