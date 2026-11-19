# Post hoc analyses for the revision (review round 1, geographies-4643561)

Written 2026-11-01, **before any of these analyses is computed**. Its SHA-256 is recorded in
`spec_revision_posthoc_2026-11-01.sha256` before the first computation. Every analysis here is post hoc: it was
specified after the results of the submitted manuscript had been read and after the reviewer reports were received.
None of them amends the pre-declaration of 1 October 2026, and none changes any locked input, test-point file or
pre-declared result. Outputs go only to `Paper_D/08_revision_posthoc/outputs/`.

Inputs: the locked inputs of `exp_map_stability/00_predeclaration/input_lock_2026-10-01.json` (verified by SHA-256
before use), the 270 training-background coordinate files `exp_map_stability/02_data/runs/lopo/*_bg.npy`, the 90
prediction maps `exp_map_stability/02_data/maps/*.npy`, the 18 seed-42 map models, and the functions of the locked
runner `run_map_stability.py` (imported, not edited). Learner settings, feature order, grid, valid mask, province
raster and SSIM settings are those of the runner.

## P1. Held-out AUC against a test background near the firms (Reviewer 1, R1-1)

Purpose: check whether discrimination stays flat across radii when the test background lies close to existing firms,
instead of across the whole held-out province.

1. **Local test backgrounds.** For each held-out province k, two new shared test backgrounds are drawn, with the same
   procedure as `make_test_points.py` except the distance band: candidates uniform in the bounding box of the province
   polygon, kept if inside the polygon, at least 500 m from every one of the 2859 presences, and no more than D from
   the nearest presence, with D = 1 km (`near1km`) and D = 5 km (`near5km`); covariates extracted from the locked
   rasters as in `make_test_points.py`; the first n_test complete candidates in draw order are kept, n_test = number of
   presences recorded in the province. Generator: `numpy.random.default_rng(20261101)` per province, budget
   B = max(200000, 2000 × n_test) drawn at once. A province that cannot be filled is reported as incomplete and its
   fold is not scored for that background.
2. **Refit.** For each of the 270 units (9 folds × 6 radii × 5 draws), the training set is rebuilt from the training
   presences of the fold and the unit's saved background coordinates (`*_bg.npy`), with covariates extracted by the
   runner's `Data.extract`. The three learners are refitted with the runner's `make()`.
3. **Reproduction check.** Each refitted model is scored on the original shared test set. Its AUC must equal the stored
   AUC in `lopo_metrics_by_fold.csv` to within 1e-9. Every mismatch is reported; a unit that fails the check is still
   scored, and the failure is stated with the results.
4. **Scores.** AUC of each refitted model on `near1km` and `near5km`. Summary as in the manuscript: median over the five
   draws of the median over the nine folds, per algorithm and radius, with the range across folds.
5. **Reading.** Report the AUC levels and whether the 1 km ring still scores below the wider rings. No significance test.

## P2. Six presences outside the polygon of their recorded province (Reviewer 1, R1-10)

1. The six presences are identified as in the existing post hoc table: point not within the polygon of its recorded
   province (`FPROVNAME`).
2. Two variants of the presence set: **reassign** (each of the six moved to the province whose polygon contains it) and
   **exclude** (the six removed, 2853 presences).
3. For each variant, all 270 units are rerun with the runner's `sample_bg` (same seeds, same budget, same ring rule,
   training presences and polygon of the variant), the three learners are refitted, and AUC is computed on the
   original shared test backgrounds with the variant's test presences. Thresholds, TSS, kappa, permutation and
   group removal are not recomputed.
4. Report the same AUC summary as the manuscript, and the change from the submitted values per algorithm and radius.

## P3. Structure term of SSIM (Reviewer 1, R1-2)

For every comparison and draw in `map_similarity_by_province.csv` (across radii against 10 km, and across algorithm
pairs), on the percentile-rank maps (runner U6) and the 7 × 7 window: the SSIM structure term
s = (σ_ab + C3) / (σ_a σ_b + C3), C3 = C2 / 2, C2 = (0.03 × 1)², with local means, variances and covariance from the
same uniform window and sample-covariance convention as skimage's `structural_similarity`, averaged over the cells of
each province whose whole window is valid (runner U5). Summary: median over draws, then median and range over the nine
provinces, as for rank SSIM.

## P4. Within-algorithm, between-draw agreement at 10 km (Reviewer 1, R1-3)

For each algorithm at 10 km, every pair of the five draws (10 pairs): raw SSIM, rank SSIM (7 × 7) and top-10% Jaccard
per province, computed with the runner's functions. Summary: median over the 10 pairs, then median and range over
provinces. This is the baseline for the between-algorithm values (which compare algorithms within a draw).

## P5. SHAP across all five draws, and normalized shares (Reviewer 2, R2-3; Reviewer 1, R1-4)

1. Map models for draws 0, 1, 7 and 2024 are refitted with the runner's `run_maps` procedure (same `sample_bg` call,
   seed and n_bg = 2859). Reproduction check: each refitted model's predictions on the valid cells must equal the
   stored map `*_s{seed}.npy` to within 1e-6; mismatches are reported.
2. SHAP values are computed exactly as runner U9 (same 1000 explained cells, same 100 reference cells, TreeSHAP
   interventional on the probability scale for RF and XGBoost, the permutation explainer with the same seed and
   max_evals for logistic regression). The seed-42 values on disk are reused.
3. Per learner × radius × draw: mean |SHAP| per covariate, its rank among the 11 covariates, and the **normalized share**
   = mean |SHAP| of the covariate / sum of mean |SHAP| over the 11 covariates.
4. Report, over the 90 learner × radius × draw combinations: how often slope, distance to road and each other covariate
   ranks in the top three; and the normalized shares by algorithm (median over radii and draws). The group-removal
   test is not repeated (stated as first-draw only).

## P6. Variance components of hotspot membership (Reviewer 1, R1-5)

On the same cells and indicator I_rld(c) as Equation (18) (configuration-dependent cells, with and without the 1 km
ring), per cell and pooled over cells per province and over the study area:

1. Sums of squares for R, L, D, RL, RD, LD and RLD (one observation per combination), degrees of freedom per cell
   R−1, L−1, D−1, (R−1)(L−1), (R−1)(D−1), (L−1)(D−1), (R−1)(L−1)(D−1), multiplied by the number of cells when pooled,
   and mean squares MS = SS / df.
2. Variance components from expected mean squares, radius and algorithm fixed, draw random, RLD as the residual:
   σ²_D = (MS_D − MS_RLD) / (R·L), σ²_RD = (MS_RD − MS_RLD) / L, σ²_LD = (MS_LD − MS_RLD) / R, σ²_res = MS_RLD;
   for the fixed terms θ²_R = (MS_R − MS_RD) / (L·D), θ²_L = (MS_L − MS_LD) / (R·D), θ²_RL = (MS_RL − MS_RLD) / D.
   Negative estimates are set to 0. Shares = each component / sum of all components.
3. Report MS, the components and their shares beside the raw SS shares already reported.

## P7. Visually confirmed and unchecked presences (Reviewer 2, R2-1)

Descriptive only, no refit: per province, the counts of the 731 visually confirmed and 2128 unchecked presences; the
median of each covariate in each group; the distance from each unchecked presence to the nearest confirmed presence;
and, for the seed-42 maps at 10 km, the median predicted score at confirmed and at unchecked locations and the share of
each group lying in the study-wide top 10%.

## Reporting rules

All results are labeled post hoc in the manuscript and the response letter. Whatever they show is reported, including
results that weaken a submitted statement; in that case the statement is revised. No analysis is added, dropped or
changed after this file is hashed without a dated note appended below.

## Notes appended after hashing

### Note 1 (2026-11-01, after P1, P3–P7 were computed and while P2 was running)

This note changes no analysis. It records terminology and reporting choices made before any result is written into
the manuscript. At the time of this note, P1, P3, P4, P5, P6 and P7 had been computed under the specification as
originally hashed (SHA-256 7434b645295c28cffc7d33e4eff1ba8642353017a3a21da7808d71ace7c9082a, kept unchanged as
`spec_revision_posthoc_2026-11-01_v1_as_hashed.md`), and P2 was running. The note is therefore not a pre-computation
revision and is not described as one.

1. **P1 refit.** The 810 fold-level models of the experiment were not saved (only the 18 seed-42 map models were), so
   P1 had to refit them. All 810 refitted models reproduced the stored held-out AUC (maximum absolute difference
   1.1e-16), so the refits are the original models.
2. **Number of fits in P2.** A unit is one fold × radius × draw (9 × 6 × 5 = 270), and each unit fits the three
   algorithms: 810 fits per variant, 1620 fits for the reassigned and excluded variants together. Both variants are
   run in full; neither is a reduced sensitivity check.
3. **P6 outcome and wording.** The outcome is the per-cell hotspot status I_rld(c) (study-wide top 10% of each map),
   as in Equation (18), not the predicted score. The results are reported as a **descriptive variance decomposition**
   (mean squares and expected-mean-square shares). Neighboring cells are spatially autocorrelated and are not
   independent observations, so no inferential statement (test, interval or p-value) is attached.
4. **Results to report.**
   - P1: median AUC per algorithm and radius for each test background, the provincial range, and the difference from
     the original test background.
   - P2: ΔAUC from the submitted values per province and the summary median, for both the reassigned and the excluded
     variant.
   - P3: structure-term SSIM beside the full rank SSIM.
   - P4: within-algorithm between-draw SSIM and Jaccard at 10 km beside the between-algorithm values at 10 km.
   - P5: median and range across the five draws of mean |SHAP| and of the normalized share, and the number of the 90
     algorithm × radius × draw combinations in which slope and distance to road rank in the top three.
   - P6: mean squares and decomposition shares beside the raw SS shares.
   - P7: counts and proportions of visually confirmed and unchecked presences per province, with the descriptive
     comparisons. Unchecked records are not interpreted as misplaced.

### Note 2 (2026-11-03, after all analyses were computed)

This note changes no computation. After submission, all 2,859 facility coordinates were visually inspected in Google
Earth by an inspector external to the study (author statement, 2026-11-03), every coordinate was judged to correspond
plausibly to a visible industrial facility or developed site, and no coordinate was changed. The split into 731 records
inspected before submission and 2,128 not individually checked therefore no longer describes the data, and P7 (the
descriptive comparison of the two groups) is no longer reported in the manuscript or the Supplementary Materials. Its
outputs are kept unchanged in `outputs/P7_confirmed_vs_unchecked.csv`. P1 to P6 are reported as before.
