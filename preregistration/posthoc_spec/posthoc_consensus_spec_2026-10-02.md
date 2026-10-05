# Post-hoc specification: consensus core and configuration-dependent hotspot areas

**Status:** POST-HOC. Written 2 October 2026, after the map-stability run outputs (readout v2, seed-vs-learner
note) had been read, but **before** this analysis was computed. It is not part of the locked confirmatory
analysis in `exp_map_stability/00_predeclaration/`. Its SHA-256 is recorded in
`posthoc_consensus_spec_2026-10-02.sha256` before the script is run; any later change will be appended
as a dated amendment, never edited in place.

## Inputs (read-only, from `exp_map_stability/02_data/maps/`)

- 90 prediction arrays `{RF,XGB,LR}_{1km,5km,10km,20km,25km,Rinf}_s{42,0,1,7,2024}.npy`
  (one value per valid 500 m cell, 382,056 cells).
- `valid_mask.npy`, `province_raster.npy` (index i+1 = i-th province of the locked boundary).
- Province code order is recovered, not assumed: per-province top-10% counts recomputed from
  `RF_1km_s42` and `RF_10km_s42` must equal `n_top10_a` / `n_top10_b` in `map_similarity_by_province.csv`
  exactly; otherwise the script stops.
- SHA-256 of every input is checked against `exp_map_stability/04_notes/results_manifest_2026-10-01.json`.

No model is fitted, no prediction is recomputed, nothing is written into `exp_map_stability/` or `paper_C/`.

## Definitions (fixed now)

1. **Hotspot membership** `I[r,l,s](cell)` = 1 if the cell's score >= the 90th percentile of that map
   over all valid cells (identical to locked rule U7: study-wide, per map, no rescaling).
2. **Consensus frequency** `f(cell)` = mean of `I` over the maps in the set.
3. **Two map sets, both reported:**
   - **Set A (primary): all 90 maps** (6 radii x 3 learners x 5 seeds).
   - **Set B (sensitivity): 75 maps excluding 1 km**, because 1 km has lower held-out discrimination
     (readout v2). Set B is a sensitivity view, not a correction; Set A is not replaced.
4. **Classes** (labels chosen to avoid the word "uncertainty", which the framework reserves against):
   - `consensus core`: f >= 0.9
   - `configuration-dependent`: 0 < f < 0.9
   - `never hotspot`: f = 0
   The full distribution of f is also reported, so the 0.9 cut can be read against the whole curve.
5. **Which choice moves a cell** — descriptive sum-of-squares decomposition of the binary `I` over the
   balanced 6 x 3 x 5 design, per cell, summed over configuration-dependent cells, per province and
   corridor-wide: SS_radius, SS_learner, SS_seed (main effects) and SS_rest (all interactions). Reported
   as shares of SS_total. For Set B the design is 5 x 3 x 5.

## Reporting rules

- Area in km2 = cells x 0.25. Per province and corridor-wide.
- Descriptive only: no p-values, no significance language, pixels are not replicates.
- Seeds are five background draws reusing the same presences; they are not independent datasets.
- Output wording: "consensus core" / "configuration-dependent" / "never hotspot". Not "uncertain",
  "confidence", or "certain".
- The 1 km result enters Set A unchanged; no ring or learner is dropped from Set A after seeing results.

## Outputs (all under `Paper_D/01_posthoc_consensus/`)

`posthoc_consensus.py`, `consensus_freq_setA.npy`, `consensus_freq_setB.npy`,
`consensus_by_province.csv`, `ss_decomposition_by_province.csv`, `f_distribution.csv`,
`fig_consensus_map.png/.pdf`, `posthoc_consensus_readout_2026-10-02.md`, `outputs_sha256.json`.
