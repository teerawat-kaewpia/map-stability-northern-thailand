import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, '..');
const outPath = path.join(root, '04_notes', 'results_readout_2026-10-01_v2.md');

function parseCsv(text) {
  const rows = [];
  let row = [], value = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { value += '"'; i++; }
      else if (c === '"') quoted = false;
      else value += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(value); value = ''; }
    else if (c === '\n') { row.push(value.replace(/\r$/, '')); rows.push(row); row = []; value = ''; }
    else value += c;
  }
  if (value.length || row.length) { row.push(value.replace(/\r$/, '')); rows.push(row); }
  const header = rows.shift().map(x => x.replace(/^\uFEFF/, ''));
  return rows.filter(r => r.length > 1).map(r => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}
const readCsv = async rel => parseCsv(await readFile(path.join(root, rel), 'utf8'));
const n = x => Number(x);
const quantile = (xs, p) => {
  const a = [...xs].sort((x, y) => x - y), pos = (a.length - 1) * p;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
};
const median = xs => quantile(xs, 0.5);
const fmt = x => Number(x).toFixed(3);
const statText = (rows, metric) => {
  const vals = rows.map(r => n(r[metric])).filter(Number.isFinite);
  if (!vals.length) return 'NA';
  return `${fmt(median(vals))} [${fmt(Math.min(...vals))}–${fmt(Math.max(...vals))}]`;
};

const lopo = await readCsv('02_data/runs/lopo_fold_median_across_seeds.csv');
const spread = await readCsv('02_data/runs/lopo_fold_spread_by_seed.csv');
const map = await readCsv('02_data/runs/map_similarity_summary_across_seeds.csv');
const mapByProvince = await readCsv('02_data/runs/map_similarity_by_province.csv');
const shap = await readCsv('02_data/runs/shap/shap_summary.csv');
const status = await readCsv('02_data/runs/lopo_unit_status.csv');
const shapCoverage = JSON.parse(await readFile(path.join(root, '02_data/runs/shap/shap_summary_coverage.json'), 'utf8'));
const radii = ['1km', '5km', '10km', '20km', '25km', 'Rinf'];
const learners = ['RF', 'XGB', 'LR'];
if (status.length !== 270 || new Set(status.map(x => `${x.fold}|${x.radius}|${x.seed}`)).size !== 270 || status.some(x => x.status !== 'complete')) {
  throw new Error('LOPO status table is not 270 unique complete units.');
}
if (map.length !== 1485 || map.some(x => Number(x.n_seeds_compared) !== 5)) throw new Error('Map summary does not have full five-seed coverage.');
if (mapByProvince.length !== 7425 || mapByProvince.some(x => x.status !== 'ok' || !Number.isFinite(Number(x.value)))) {
  throw new Error('Province-level map comparisons contain a missing, non-finite, or non-ok row.');
}
if (shap.length !== 198 || shapCoverage.expected !== 18 || shapCoverage.present !== 18 || shapCoverage.missing.length) {
  throw new Error('SHAP outputs do not cover all 18 configurations and 198 features.');
}
function npyFloatInfo(buf) {
  if (![0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59].every((v, i) => buf[i] === v)) throw new Error('Invalid NPY magic.');
  const major = buf[6], headStart = major === 1 ? 10 : 12;
  const headerLength = major === 1 ? buf.readUInt16LE(8) : buf.readUInt32LE(8);
  const dataOffset = headStart + headerLength;
  const header = buf.toString('latin1', headStart, dataOffset);
  const dtype = header.match(/'descr':\s*'([^']+)'/)?.[1];
  const shapeText = header.match(/'shape':\s*\(([^)]*)\)/)?.[1] ?? '';
  const shape = [...shapeText.matchAll(/\d+/g)].map(x => Number(x[0]));
  if (!['<f4', '<f8'].includes(dtype) || shape.length !== 1 || shape[0] !== 382056) throw new Error(`Invalid prediction-array header: ${dtype}, ${shape}`);
  return { dataOffset, bytesPerValue: dtype === '<f4' ? 4 : 8 };
}
let predictionValueCount = 0, predictionMin = Infinity, predictionMax = -Infinity;
for (const file of (await (await import('node:fs/promises')).readdir(path.join(root, '02_data/maps')))
  .filter(name => /^(RF|XGB|LR)_.+_s\d+\.npy$/.test(name))) {
  const buffer = await readFile(path.join(root, '02_data/maps', file));
  const { dataOffset, bytesPerValue } = npyFloatInfo(buffer);
  for (let offset = dataOffset; offset < buffer.length; offset += bytesPerValue) {
    const value = bytesPerValue === 4 ? buffer.readFloatLE(offset) : buffer.readDoubleLE(offset);
    if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error(`Invalid prediction value in ${file}: ${value}`);
    predictionMin = Math.min(predictionMin, value); predictionMax = Math.max(predictionMax, value); predictionValueCount++;
  }
}
if (predictionValueCount !== 90 * 382056) throw new Error(`Checked ${predictionValueCount} prediction values; expected ${90 * 382056}.`);

const lopoRows = [];
for (const radius of radii) for (const learner of learners) {
  const r = lopo.find(x => x.radius === radius && x.learner === learner);
  if (!r) throw new Error(`Missing LOPO summary ${radius}/${learner}`);
  const val = (metric, suffix) => `${fmt(r[`${metric}_median_${suffix}`])}`;
  const range = metric => `${val(metric, 'median')} (${val(metric, 'min')}–${val(metric, 'max')})`;
  lopoRows.push(`| ${radius} | ${learner} | ${range('auc')} | ${range('tss')} | ${range('kappa')} |`);
}

const mapMetrics = ['ssim_raw_7', 'ssim_rank_7', 'jaccard_top10'];
const mapSensitivityMetrics = ['ssim_raw_21', 'ssim_rank_21'];
const comparisonRows = (kind, build, metrics = mapMetrics) => {
  const lines = [];
  for (const spec of build) {
    const d = map.filter(x => x.comparison === kind && spec.match(x));
    if (!d.length || d.some(x => Number(x.n_seeds_compared) !== 5)) throw new Error(`Incomplete map comparison: ${spec.label}`);
    lines.push(`| ${spec.label} | ${metrics.map(m => statText(d.filter(x => x.metric === m), 'median')).join(' | ')} |`);
  }
  return lines;
};
const radiusSpecs = [];
for (const learner of learners) for (const radius of radii.filter(x => x !== '10km')) {
  radiusSpecs.push({ label: `${learner}: ${radius} vs 10 km`, match: x => x.within === learner && x.a === `${learner}_${radius}` && x.b === `${learner}_10km` });
}
const learnerPairs = [['RF', 'XGB'], ['RF', 'LR'], ['XGB', 'LR']];
const learnerSpecs = [];
for (const radius of radii) for (const [a, b] of learnerPairs) {
  learnerSpecs.push({ label: `${radius}: ${a} vs ${b}`, match: x => x.within === radius && x.a === `${a}_${radius}` && x.b === `${b}_${radius}` });
}

const shapFeatures = [...new Set(shap.map(x => x.feature))];
const shapTop3 = Object.fromEntries(shapFeatures.map(f => [f, 0]));
const shapDirections = Object.fromEntries(shapFeatures.map(f => [f, { positive: 0, negative: 0 }]));
const shapRows = [];
for (const learner of learners) {
  const byFeature = shapFeatures.map(feature => {
    const rows = shap.filter(x => x.learner === learner && x.feature === feature);
    if (rows.length !== radii.length) throw new Error(`Incomplete SHAP cells: ${learner}/${feature}`);
    return { feature, value: median(rows.map(x => n(x.mean_abs_shap))) };
  }).sort((a, b) => b.value - a.value);
  for (const row of byFeature.slice(0, 4)) shapRows.push(`| ${learner} | ${row.feature} | ${fmt(row.value)} |`);
  for (const radius of radii) {
    const ranked = shap.filter(x => x.learner === learner && x.radius === radius)
      .sort((a, b) => n(b.mean_abs_shap) - n(a.mean_abs_shap));
    for (const row of ranked.slice(0, 3)) shapTop3[row.feature]++;
  }
}
for (const feature of shapFeatures) {
  for (const row of shap.filter(x => x.feature === feature)) {
    const value = n(row.direction_spearman);
    if (value > 0) shapDirections[feature].positive++;
    else if (value < 0) shapDirections[feature].negative++;
  }
}

const perm = {}, lofo = {};
for (const row of status) {
  if (row.status !== 'complete') throw new Error(`Non-complete LOPO unit ${row.fold}/${row.radius}/${row.seed}`);
}
for (const row of status) {
  const rel = `02_data/runs/lopo/${row.fold}_${row.radius}_s${row.seed}.json`;
  const unit = JSON.parse(await readFile(path.join(root, rel), 'utf8'));
  for (const learner of learners) {
    const record = unit.learners[learner];
    for (const [key, pair] of Object.entries(record.perm_auc_drop)) (perm[key] ??= []).push(Number(pair[0]));
    for (const [key, value] of Object.entries(record.lofo_auc_drop ?? {})) (lofo[key] ??= []).push(Number(value));
  }
}
const importanceTable = (obj, limit) => Object.entries(obj).map(([feature, values]) => ({
  feature, n: values.length, median: median(values), lo: quantile(values, 0.25), hi: quantile(values, 0.75),
})).sort((a, b) => b.median - a.median).slice(0, limit);

const permRows = importanceTable(perm, 12).map(x => `| ${x.feature} | ${x.n} | ${fmt(x.median)} | ${fmt(x.lo)}–${fmt(x.hi)} |`);
const lofoRows = importanceTable(lofo, 5).map(x => `| ${x.feature} | ${x.n} | ${fmt(x.median)} | ${fmt(x.lo)}–${fmt(x.hi)} |`);
const shapTopRows = Object.entries(shapTop3).sort((a, b) => b[1] - a[1]).filter(([, count]) => count > 0)
  .map(([feature, count]) => `| ${feature} | ${count}/18 | ${shapDirections[feature].positive} positive / ${shapDirections[feature].negative} negative |`);

const auc1 = lopo.filter(x => x.radius === '1km').map(x => n(x.auc_median_median));
const aucOther = lopo.filter(x => x.radius !== '1km').map(x => n(x.auc_median_median));
const lopoRaw = await readCsv('02_data/runs/lopo_metrics_by_fold.csv');
const maxTssKappaDifference = Math.max(...lopoRaw.map(x => Math.abs(n(x.tss) - n(x.kappa))));

const text = `# Map-stability experiment: result readout (v2)

**Version 2. Completed:** 1 October 2026, 22:22:58 (Asia/Bangkok), as reported by the operator. This corrects the SSIM window labels in v1: the locked primary is 7×7; 21×21 is sensitivity. **No models were rerun for this readout.** Values are read from saved output tables/JSON and summarized under the locked rules; they are descriptive, not significance tests. Version 1 is retained unchanged for audit history.

## Integrity and coverage

- LOPO: ${status.length}/270 fold × radius × seed units marked complete; per unit the saved JSON contains RF, XGB and LR. The reported run audit also verified backgrounds and required fields.
- Map models/prediction surfaces: 18 seed-42 model files and 90 prediction arrays (six radii × three learners × five seeds); independently checked ${predictionValueCount.toLocaleString('en-US')} array values for expected shape, finiteness and [0,1] range (observed extrema ${fmt(predictionMin)}–${fmt(predictionMax)}).
- Map comparisons: ${mapByProvince.length.toLocaleString('en-US')} province × seed rows, all status=ok with finite values; summarized to ${map.length.toLocaleString('en-US')} province-level rows, each covering all five seeds.
- SHAP: 18/18 map models, 198 feature rows. Outputs are seed-42 map-model explanations.
- Manifest: \`results_manifest_2026-10-01.json\` hashes every file in \`02_data/runs\` and \`02_data/maps\`, plus the execution provenance files and both readout versions.

## Predictive performance on held-out provinces

Each cell is the median of the five seed-specific medians across nine province folds; parentheses give the min–max of those five seed-level fold medians. These are seed ranges, not confidence intervals.

| Radius | Learner | AUC | TSS | Cohen's κ |
|---|---|---:|---:|---:|
${lopoRows.join('\n')}

**Reading:** 1-km backgrounds have lower held-out discrimination (median AUC ${fmt(Math.min(...auc1))}–${fmt(Math.max(...auc1))}) than the 5–R∞ configurations (${fmt(Math.min(...aucOther))}–${fmt(Math.max(...aucOther))}). From 5 km outward, AUC is high and differences among radii are small; there is no clear monotonic gain from increasing the radius beyond 5 km. TSS and κ are identical to numerical precision in the saved rows (maximum absolute difference ${maxTssKappaDifference.toExponential(1)}). In a 1:1 test set, chance-expected agreement for Cohen's κ is 0.5 regardless of predicted-positive prevalence; κ therefore equals sensitivity + specificity − 1 (TSS) and adds no independent metric information here.

## Similarity across radii, within the same learner

The locked primary SSIM window is 7×7 cells. For compact overview only, each entry reports the median of the nine province-level medians (each province median is over five seeds); brackets show the range across those nine provincial medians. **The province-by-province output CSV is the primary reporting source; these cross-province summaries do not replace it.** Jaccard reports top-decile hotspot overlap.

Primary results (7×7 SSIM):

| Comparison to 10 km | Raw SSIM 7×7 | Rank SSIM 7×7 | Top-10% Jaccard |
|---|---:|---:|---:|
${comparisonRows('across_radii', radiusSpecs).join('\n')}

21×21 SSIM sensitivity results:

| Comparison to 10 km | Raw SSIM 21×21 | Rank SSIM 21×21 |
|---|---:|---:|
${comparisonRows('across_radii', radiusSpecs, mapSensitivityMetrics).join('\n')}

**Reading:** At 1 km versus 10 km, median provincial Jaccard is about 0.17–0.18 across learners, and the 7×7 SSIM values quantify the accompanying map differences. Agreement at larger radii varies by learner and radius. The pre-declaration did not define a numeric cutoff for calling SSIM or hotspot overlap “low”; the coefficients are therefore reported descriptively, without classifying a comparison as passing or failing a low-agreement rule.

## Similarity between learners at the same radius

Same compact summary convention; the province-specific \`map_similarity_by_province.csv\` remains the primary result, with cross-province medians used only as an overview.

Primary results (7×7 SSIM):

| Radius and pair | Raw SSIM 7×7 | Rank SSIM 7×7 | Top-10% Jaccard |
|---|---:|---:|---:|
${comparisonRows('across_learners', learnerSpecs).join('\n')}

21×21 SSIM sensitivity results:

| Radius and pair | Raw SSIM 21×21 | Rank SSIM 21×21 |
|---|---:|---:|
${comparisonRows('across_learners', learnerSpecs, mapSensitivityMetrics).join('\n')}

At 10 km, the three learner AUC medians span ${fmt(Math.min(...lopo.filter(x => x.radius === '10km').map(x => n(x.auc_median_median))))}–${fmt(Math.max(...lopo.filter(x => x.radius === '10km').map(x => n(x.auc_median_median))))} (a difference of ${fmt(Math.max(...lopo.filter(x => x.radius === '10km').map(x => n(x.auc_median_median))) - Math.min(...lopo.filter(x => x.radius === '10km').map(x => n(x.auc_median_median))))}); median provincial Jaccard summaries across learner pairs span ${fmt(Math.min(...map.filter(x => x.comparison === 'across_learners' && x.within === '10km' && x.metric === 'jaccard_top10').map(x => n(x.median))))}–${fmt(Math.max(...map.filter(x => x.comparison === 'across_learners' && x.within === '10km' && x.metric === 'jaccard_top10').map(x => n(x.median))))}. This describes observed agreement; the pre-declaration set no threshold for deeming these overlaps “low.” Similarity is province-specific, so read the province-level CSV rather than treating these summaries as a uniform corridor-wide pattern.

## SHAP and factor-importance summaries

### SHAP

Median mean absolute SHAP over the six radii, by learner (probability scale; larger values indicate greater contribution magnitude in these fitted models, not significance):

| Learner | Feature | Median mean \\|SHAP\\| |
|---|---|---:|
${shapRows.join('\n')}

Features appearing in the top three mean-absolute-SHAP ranks across the 18 learner × radius models:

| Feature | Configurations in top three | Direction sign of value–SHAP Spearman across 18 configs |
|---|---:|---|
${shapTopRows.join('\n')}

Slope and distance to road recur most often in the top three. Nightlight and forest also recur, while the leading feature and SHAP magnitude differ by learner. Direction signs are descriptive correlations, not causal effects. Tree SHAP uses the locked interventional reference; LR uses the locked sampled PermutationExplainer, so LR SHAP is an approximation. These summaries compare explanatory allocation, not a hypothesis test that SHAP differs between models.

### Permutation importance and LOFO (supporting)

Permutation rows pool the 9 folds × 6 radii × 5 seeds × 3 learners (810 unit-level values per feature); values are median AUC drop and central interquartile range. Joint built-up + nightlight permutation applies the same row shuffle to both variables.

| Feature/group | n | Median AUC drop | Approx. IQR |
|---|---:|---:|---:|
${permRows.join('\n')}

LOFO rows pool 9 folds × 6 radii × 3 learners at seed 42 (162 values per group); conclusions are conditional on that background draw.

| Group removed | n | Median AUC drop | Approx. IQR |
|---|---:|---:|---:|
${lofoRows.join('\n')}

These are model-dependence diagnostics, not causal effects or p-values. Correlated covariates can share or mask importance; the grouped built-up/nightlight permutation helps expose their joint information but does not isolate either variable.

## Interpretation limits

- The five seeds vary training-background draws; they are not five independent datasets. The test-background set is fixed and paired across model conditions.
- Province-level estimates are based on nine held-out provinces. The pre-declaration records registry-province/polygon mismatches, so this is not a perfectly geographic separation at every boundary.
- High AUC does not establish that a suitability map is spatially stable or preferable. Map similarity and discrimination answer different questions; do not select a radius or learner from AUC alone.
- The SHAP and importance results describe fitted models under this experiment's sampling and covariates. They do not prove causal drivers, legal suitability, or statistical significance of between-model differences.

## Source outputs

- \`02_data/runs/lopo_fold_median_across_seeds.csv\` and \`lopo_fold_spread_by_seed.csv\`
- \`02_data/runs/map_similarity_summary_across_seeds.csv\` and \`map_similarity_by_province.csv\`
- \`02_data/runs/shap/shap_summary.csv\` and \`shap_summary_coverage.json\`
- \`02_data/runs/lopo/*.json\` (fold/radius/seed model metrics, permutation and LOFO)
- \`02_data/maps/\` (prediction and seed-42 difference arrays)
`;

await writeFile(outPath, text, 'utf8');
console.log(JSON.stringify({ path: path.relative(root, outPath), status_units: status.length,
  map_similarity_rows: map.length, shap_rows: shap.length, perm_unit_values_per_feature: 810,
  lofo_unit_values_per_group: 162, prediction_values_checked: predictionValueCount,
  prediction_min: predictionMin, prediction_max: predictionMax }, null, 2));
