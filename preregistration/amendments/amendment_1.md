<!-- extracted from predeclaration_2026-10-01_redacted.md -->
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
