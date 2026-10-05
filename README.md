# Comparable Accuracy, Divergent Hotspots: replication package

Replication package for the article:

Kaewpia, T.; Changruenngam, T.; Suantai, S.; Prommai, T.; Klinhnu, J. *Comparable Accuracy, Divergent Hotspots: A Pre-specified Crossed Experiment on Background Radius and Algorithm Choice in Industrial-Location Suitability Mapping, Northern Thailand.* Submitted to *Geographies* (MDPI).

**Status:** submitted. **DOI:** to be assigned by Zenodo on publication.

The experiment crosses six background radii (1, 5, 10, 20, 25 km and R∞) with three algorithms (random forest, XGBoost, logistic regression), each fitted under five pre-declared background draws (seeds 42, 0, 1, 7, 2024). It evaluates them by leave-one-province-out validation over nine provinces of upper Northern Thailand, on shared test sets, and maps each configuration on a common 500 m grid (382,056 valid cells, WGS 84 / UTM zone 47N, EPSG:32647).

This package contains only material that can be redistributed. Several inputs are restricted (see *Available and not available*), so the full experiment cannot be re-run from this package alone, but every number reported in the article can be checked against the released tables.

## Folder structure

```text
README.md, LICENSE, CITATION.cff, environment.yml, environment_lock.txt
code/
  experiment/            make_test_points.py, run_map_stability.py, figure scripts, Node.js manifest/readout scripts
  paper_tables_figures/  manuscript figures (Figure 1: make_fig_study_area.py) and main/supplementary tables
  posthoc_consensus/     posthoc_consensus.py (exploratory consensus and variance shares)
  posthoc_sensitivity/   posthoc_sensitivity.py (post hoc sensitivity analyses, Section 3.6)
derived_data/
  tables/                fold-level metrics, summaries across draws, map similarity, unit status,
                         lopo_units/ (270 JSON units), shap/ (SHAP summary)
  consensus/             consensus-core and configuration-dependent areas, hotspot-frequency distribution,
                         variance shares by province
  posthoc_sensitivity/   post hoc tables (paired AUC differences, cut-offs, interaction, train-test distances,
                         registry/polygon mismatches) and the fold map
test_points/             test_points_release.csv, test_points_manifest.csv
preregistration/
  pre_declaration/       pre-declaration (with Amendments 1-4) and input lock, redacted
  amendments/            Amendments 1-4 extracted for convenience
  posthoc_spec/          specifications of the post hoc analyses and their SHA-256, written before computing
integrity/               input_hashes.csv, output_hashes.csv, release_manifest.csv, chronology.csv, run_logs/
```

## Installing the environment and running the scripts

```bash
conda env create -f environment.yml      # key packages; environment_lock.txt is the full lock
conda activate map-stability
```

The `.mjs` scripts need Node.js. The scripts are the ones that were run and contain the original absolute paths, which must be adapted. With the restricted inputs (firm locations, forest layer and planned-railway raster), the order is:

1. `code/experiment/make_test_points.py`
2. `code/experiment/run_map_stability.py`
3. `code/posthoc_consensus/posthoc_consensus.py`
4. `code/posthoc_sensitivity/posthoc_sensitivity.py`
5. the scripts in `code/paper_tables_figures/`

## What can be reproduced from this package

From the released files alone, without the restricted inputs:

- discrimination (AUC, TSS, kappa, thresholds) for all 810 fold-level fits and their summaries (Figure 3, Tables S2 and S4);
- map agreement, SSIM and top-10% Jaccard, by province and draw (Figures 4 and 6, Tables S3 and S5-S8);
- factor reliance: permutation and group-removal losses and SHAP summaries (Figures 7-9, Tables S9-S12);
- the consensus areas, hotspot-frequency distribution and variance shares (Table S13 and the statistics of Figure 10; the map panels need the prediction surfaces);
- all post hoc sensitivity results (Section 3.6, Tables S14-S20, Figure S1);
- the validation workflow on the shared test points.

Not reproducible from this package: Figure 1 and Table S1 (firm locations), Figure 5 and the map panels of Figure 10 (prediction surfaces), and re-fitting the models.

## Available and not available

| Item | Status | Reason |
| --- | --- | --- |
| Code, specifications, amendments, hashes | Released | The workflow can be followed and checked |
| Result tables and post hoc outputs | Released | They support the numbers in the article |
| Test points with restricted values removed | Released | The validation workflow can be checked |
| Firm coordinates and registry records (DIW) | Not released | Restricted registry data |
| Royal Forest Department forest layer and the `forest` value | Not released | Supplied for research use; redistribution restricted |
| Planned-railway alignment, `dist_railway_new_km.tif` and railway distances | Not released | The source Google My Maps map states no license |
| Prediction surfaces, consensus maps and per-cell SHAP | Not released | Derived from the restricted forest layer |
| Training-background point files and fitted models | Not released | The 1 km ring lies within 1 km of firms and would reveal their locations |

## Data restrictions and how to request access

- **Firm locations.** The 2859 presences come from the factory registry of the Department of Industrial Works (DIW), Ministry of Industry, Thailand. Access may be requested from the department.
- **Forest layer.** The Royal Forest Department forest-area condition data for B.E. 2568 may be obtained from the department. The spatial outputs derived from it (90 prediction surfaces, consensus and hotspot-frequency maps, per-cell SHAP values) will be released only with the department's permission.
- **Planned railway.** The alignment is the one published by the State Railway of Thailand on Google My Maps (https://www.google.com/maps/d/viewer?mid=1REMriQY0BxNPmksruPXyT867ewA; published 2017, exported 9 September 2025): the final alignment (323 km) and a 25.5 km branch toward Chiang Saen. OpenStreetMap features of the line mapped as under construction coincide with the main alignment but were not used to build the covariate. The distance raster would let the line be traced at 92.8 m, so it is not released, and the distance values have been removed from the test points.
- **Registry identifiers.** To protect restricted registry information, the publicly released pre-declaration redacts six factory registration identifiers. This redaction does not alter the study design, amendments, analyses, or reported results. The six firms are those whose registry province differs from the polygon they lie in; each ID is replaced by `[registry ID n redacted]`.

## Test points

`test_points/test_points_release.csv` has the columns `test_id`, `ADM1_PCODE`, `FPROVNAME` (province name in Thai), `draw_index` (order in the province's fixed random stream), `easting` and `northing` (m, EPSG:32647), nine of the eleven covariates at the point and `presence` (always 0, since all rows are background). Every `dist_*` column is in km, `slope_deg` is in degrees, `nightlight` is the value of the NPP-VIIRS-like annual composite, and `builtup` is binary. Points were drawn at least 500 m from every firm.

Three columns have been removed: `dist_nearest_presence_m` (distances to firms from many points would allow firm locations to be reconstructed), `forest` (the restricted forest layer) and `dist_railway_new_km` (see *Planned railway* above). `test_points_manifest.csv` records, per province, the SHA-256 of the original file before these columns were removed.

## Pre-declaration and chronology

The pre-declaration is an internal analysis plan. It was not deposited with an external registry before the run, and its hashes were recorded locally, so the experiment is described as pre-specified, not preregistered. The order of events can be checked against `integrity/chronology.csv` (local file dates and SHA-256 of the pre-declaration, input lock, test points, scripts, run logs and manifests), the run logs (the analysis script reports at 17:04:03 on 1 October 2026 that it verified the locked inputs, the test-point files and the pre-declaration before the first fit) and the output manifest. These dates are local file-system dates, not external timestamps.

The pre-declaration is kept verbatim apart from the redaction, so it keeps the working labels of the authors' files: "field-confirmed" refers to the 731 registry records whose coordinates were checked one by one in Google Earth by a human interpreter (the article calls them **visually confirmed**; no field GPS survey was made), and "learner" means algorithm. Some passages and file paths in the pre-declaration, the amendments, the input lock and the analysis script carry the working label of an earlier, unpublished analysis by the authors on the same data. The pre-declaration states that this experiment is separate from it: a few of its scripts are reused for the Chiang Khong distance covariate, the background sampling and the spatial-block grid (listed in the input lock), but none of its results are used.

## Checking file integrity

`integrity/release_manifest.csv` lists the SHA-256 of every file in this package (`path,bytes,sha256`). To check it:

```bash
python -c "import csv,hashlib; bad=[r['path'] for r in csv.DictReader(open('integrity/release_manifest.csv')) if hashlib.sha256(open(r['path'],'rb').read()).hexdigest()!=r['sha256']]; print('all files match' if not bad else bad)"
```

`integrity/input_hashes.csv` gives the SHA-256 of every locked input, with a flag for whether it is in this release, and `integrity/output_hashes.csv` that of the 711 experiment outputs (including the 90 prediction maps not released here), of the original (unredacted) pre-declaration files and of the post hoc outputs. A file released later can be checked against these hashes. The redacted pre-declaration and input lock do not match the hashes recorded at lock time; the hashes of the originals are in `integrity/output_hashes.csv` (source "experiment provenance (original, unredacted)").

## Licenses

- Code (`code/`): MIT License.
- Derived tables, pre-registration documents and integrity records: CC BY 4.0.
- `test_points/test_points_release.csv`: Open Database License (ODbL) 1.0. It contains values derived from OpenStreetMap, © OpenStreetMap contributors.
- The `slope_deg` values in `test_points/test_points_release.csv` are derived from the SRTM 90m Digital Elevation Database v4.1 (Jarvis et al., 2008), CIAT / CGIAR-CSI, which is acknowledged here as their source. The SRTM data themselves are not redistributed; CIAT does not permit their commercial use or redistribution without written permission.

See `LICENSE`.

## Citation

Please cite the article and this package; the citation metadata are in `CITATION.cff`. Corresponding author: Thanayut Changruenngam, thanayut.cha@crru.ac.th.
