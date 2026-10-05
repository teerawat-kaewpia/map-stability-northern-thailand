# Post-hoc sensitivity analyses — specification (2026-10-05)

**Status: POST HOC.** This specification was written on 2026-10-05, after all confirmatory and exploratory results had been read and reported, in response to a pre-submission review. It does not amend the pre-declaration of 2026-10-01. Every result it produces is reported as a *post hoc sensitivity* analysis. No model is refitted. All inputs are files saved by the locked run or by the 2026-10-02 consensus analysis.

Inputs (read only):
- `exp_map_stability/02_data/runs/lopo_metrics_by_fold.csv`
- `exp_map_stability/02_data/runs/map_similarity_by_province.csv`
- `exp_map_stability/02_data/maps/{RF,XGB,LR}_{1km,5km,10km,20km,25km,Rinf}_s{42,0,1,7,2024}.npy`
- `exp_map_stability/02_data/maps/valid_mask.npy`, `province_raster.npy`
- `exp_map_stability/02_data/test_points/test_points_TH*.csv`
- `RF_RP/data/processed/analysis_dataset_m3_base_clean.csv` (presences; coordinates are used in computation only and never written out)
- `Paper_D/01_posthoc_consensus/f_distribution.csv`

## S1. Paired AUC differences and aggregation order
- **Paired differences.** For each radius and each algorithm pair (RF–XGB, RF–LR, XGB–LR), compute ΔAUC = AUC_a − AUC_b within the same fold and draw. Take the median over draws per fold. Report the median, minimum and maximum across the nine folds, and the number of folds with Δ > 0.
- **Radius pairs.** The same paired differences are computed for each radius against 10 km within an algorithm.
- **Aggregation order.** For each configuration, compute the summary AUC four ways:
  - (a) median over draws of the median over folds (as reported);
  - (b) median over folds of the median over draws;
  - (c) median of all 45 fold × draw values;
  - (d) mean of all 45 values.
- Report the largest absolute difference from (a), and whether these two statements hold under every order: "from 5 km outward, 0.970–0.976" and "at 10 km, the algorithms differ by no more than 0.004".

## S2. Hotspot counts, ties and cut-off sensitivity
- **Counts.** From `map_similarity_by_province.csv`, take the 10 km algorithm pairs. For each province, report the median over draws of |H_A|, |H_B|, |H_A ∩ H_B|, |H_A ∪ H_B| and the Jaccard index.
- **Ties.** For every map at its 90th-percentile threshold (`np.quantile(v, 0.9)`, as in the run), report:
  - the number of cells equal to the threshold;
  - the excess of |H| over 10% of the valid cells.
- **Cut-off sensitivity.**
  - Recompute the hotspot sets at the top 5%, 10% and 20% (threshold = `np.quantile(v, 1 − p)`, cells ≥ threshold, study-wide per map, as in the run).
  - For the same comparisons as the run (each radius against 10 km within an algorithm; algorithm pairs at each radius), compute the provincial Jaccard. Take the median over draws per province, then the median and range across provinces.
  - The 10% values must reproduce the run's values. Any difference is reported as an audit failure.
- **Consensus cut-off.** From `f_distribution.csv`, report the area with f ≥ 0.8, f ≥ 0.9 and f = 1, for set A (90 maps) and set B (75 maps without 1 km).

## S3. Radius × algorithm interaction
- Using top-10% membership I_{r,l,d}(c) as in the consensus analysis, compute:
  - SS_RL = D · Σ_{r,l} (Ī_{rl·} − Ī_{r··} − Ī_{·l·} + Ī)²;
  - the remaining interactions, SS_T − SS_R − SS_L − SS_D − SS_RL, which all involve the draw.
- Sum each over the configuration-dependent cells (0 < f < 0.9 within the set, as in the consensus analysis) of the study area, and report it as a share of ΣSS_T, for sets A and B. Use the same cell set as the consensus analysis, so that the shares add to those already reported.

## S4. Spatial diagnostics (descriptive)
For each fold (held-out province), report:
- n test presences and n test background points;
- the distance from each test presence to its nearest training presence (presences of the other eight provinces) and the same for each test background point, as the median, 10th and 90th percentiles.
Also report the six registry/polygon mismatches recorded in the pre-declaration: registry province, polygon province, and distance to that polygon's edge. No identifiers are given.
Produce a fold map with the nine provinces (folds), each labeled with its code and presence count, without firm locations.

Outputs go to `Paper_D/06_posthoc_sensitivity/`; the SHA-256 of this file is recorded in `posthoc_sensitivity_spec_2026-10-05.sha256` before the script is run.
