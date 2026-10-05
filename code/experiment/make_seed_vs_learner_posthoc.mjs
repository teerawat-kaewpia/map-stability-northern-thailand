import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, '..');
const outCsv = path.join(root, '04_notes', 'posthoc_jaccard_seed_vs_learner_2026-10-02.csv');
const outMd = path.join(root, '04_notes', 'posthoc_seed_vs_learner_2026-10-02.md');
const mapDir = path.join(root, '02_data', 'maps');
const seeds = [42, 0, 1, 7, 2024];
const learners = ['RF', 'XGB', 'LR'];
const pairs = [['RF', 'XGB'], ['RF', 'LR'], ['XGB', 'LR']];

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

function parseNpy(buffer) {
  if (![0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59].every((v, i) => buffer[i] === v)) throw new Error('Invalid NPY magic.');
  const major = buffer[6], hs = major === 1 ? 10 : 12;
  const hl = major === 1 ? buffer.readUInt16LE(8) : buffer.readUInt32LE(8);
  const offset = hs + hl, header = buffer.toString('latin1', hs, offset);
  const dtype = header.match(/'descr':\s*'([^']+)'/)?.[1];
  const shape = [...(header.match(/'shape':\s*\(([^)]*)\)/)?.[1] ?? '').matchAll(/\d+/g)].map(x => Number(x[0]));
  return { dtype, shape, offset, data: buffer.subarray(offset) };
}

function median(values) {
  const a = [...values].sort((x, y) => x - y), m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}
const fmt = x => Number(x).toFixed(3);
const csvEscape = v => {
  const s = String(v);
  return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
};
const arrayVals = (npy, count) => {
  if (npy.dtype !== '<f4' || npy.shape.length !== 1 || npy.shape[0] !== count) throw new Error(`Unexpected map array: ${npy.dtype}/${npy.shape}`);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = npy.data.readFloatLE(i * 4);
  return out;
};

const validNpy = parseNpy(await readFile(path.join(mapDir, 'valid_mask.npy')));
const provNpy = parseNpy(await readFile(path.join(mapDir, 'province_raster.npy')));
if (validNpy.shape.length !== 2 || JSON.stringify(validNpy.shape) !== JSON.stringify(provNpy.shape)) throw new Error('Grid masks do not share a 2-D shape.');
const [height, width] = validNpy.shape, nGrid = height * width;
if (!['|b1', '|u1'].includes(validNpy.dtype) || provNpy.dtype !== '|u1' || validNpy.data.length !== nGrid || provNpy.data.length !== nGrid) throw new Error('Unexpected mask dtype or shape.');
const fullProv = new Uint8Array(nGrid), cellProv = [];
for (let i = 0; i < nGrid; i++) {
  if (validNpy.data[i]) {
    const p = provNpy.data[i];
    if (p < 1 || p > 9) throw new Error(`Invalid province code ${p} at valid cell ${i}.`);
    fullProv[i] = p; cellProv.push(p);
  }
}
if (cellProv.length !== 382056) throw new Error(`Expected 382,056 valid cells; got ${cellProv.length}.`);

const predictions = new Map();
const topMasks = new Map();
for (const learner of learners) for (const seed of seeds) {
  const key = `${learner}_10km_s${seed}`;
  const npy = parseNpy(await readFile(path.join(mapDir, `${key}.npy`)));
  const values = arrayVals(npy, cellProv.length);
  if ([...values].some(v => !Number.isFinite(v) || v < 0 || v > 1)) throw new Error(`Invalid prediction value in ${key}.`);
  const sorted = values.slice().sort();
  const pos = (sorted.length - 1) * 0.9, lo = Math.floor(pos), hi = Math.ceil(pos);
  const threshold = sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  const top = new Uint8Array(values.length);
  for (let i = 0; i < values.length; i++) top[i] = values[i] >= threshold ? 1 : 0;
  predictions.set(key, values); topMasks.set(key, top);
}

const mapCsvText = await readFile(path.join(root, '02_data', 'runs', 'map_similarity_by_province.csv'), 'utf8');
const mapRows = parseCsv(mapCsvText).filter(x => x.comparison === 'across_learners' && x.within === '10km' && x.metric === 'jaccard_top10');
if (mapRows.length !== 135 || mapRows.some(x => x.status !== 'ok')) throw new Error(`Expected 135 valid 10-km cross-learner Jaccard rows; found ${mapRows.length}.`);
const csvProvNames = [...new Set(mapRows.map(x => x.province))].sort();

function jaccard(aName, bName, provinceValue) {
  const a = topMasks.get(aName), b = topMasks.get(bName);
  let intersection = 0, union = 0;
  for (let i = 0; i < cellProv.length; i++) if (cellProv[i] === provinceValue) {
    if (a[i] && b[i]) intersection++;
    if (a[i] || b[i]) union++;
  }
  return union ? intersection / union : NaN;
}

// Match raster category values to province labels using every locked cross-learner, same-seed Jaccard.
const sigKeys = pairs.flatMap(([a, b]) => seeds.map(seed => `${a}_${seed}|${b}_${seed}`));
const csvLookup = new Map(mapRows.map(x => [`${x.a}|${x.b}|${x.seed}|${x.province}`, Number(x.value)]));
const signatureCost = Array.from({ length: 9 }, (_, ci) => csvProvNames.map(name => {
  let total = 0;
  for (const key of sigKeys) {
    const [a, b] = key.split('|');
    const [la, sa] = a.split('_'), [lb, sb] = b.split('_');
    const calculated = jaccard(`${la}_10km_s${sa}`, `${lb}_10km_s${sb}`, ci + 1);
    const observed = csvLookup.get(`${la}_10km|${lb}_10km|${sa}|${name}`);
    if (!Number.isFinite(observed) || !Number.isFinite(calculated)) throw new Error('Cannot align province raster with saved comparison table.');
    total += Math.abs(calculated - observed);
  }
  return total;
}));
let states = new Map([[0, { cost: 0, assignment: [] }]]);
for (let ci = 0; ci < 9; ci++) {
  const next = new Map();
  for (const [mask, state] of states) for (let ni = 0; ni < 9; ni++) if (!(mask & (1 << ni))) {
    const nm = mask | (1 << ni), candidate = { cost: state.cost + signatureCost[ci][ni], assignment: [...state.assignment, ni] };
    if (!next.has(nm) || candidate.cost < next.get(nm).cost) next.set(nm, candidate);
  }
  states = next;
}
const assignment = states.get((1 << 9) - 1).assignment;
const provinceNameByValue = new Map(assignment.map((i, ci) => [ci + 1, csvProvNames[i]]));
const mappingMaxError = Math.max(...signatureCost.map((row, ci) => row[assignment[ci]] / sigKeys.length));
if (mappingMaxError > 1e-9) throw new Error(`Province-raster labels did not reconcile to saved CSV (mean abs error ${mappingMaxError}).`);

const cross = [], within = [];
for (const provinceValue of Array.from({ length: 9 }, (_, i) => i + 1)) {
  const province = provinceNameByValue.get(provinceValue);
  for (const [a, b] of pairs) for (const seed of seeds) {
    cross.push({ comparison: 'between_learners_same_seed', province, pair: `${a}-${b}`, model_a: a, model_b: b,
      seed_a: seed, seed_b: seed, jaccard: jaccard(`${a}_10km_s${seed}`, `${b}_10km_s${seed}`, provinceValue) });
  }
  for (const learner of learners) for (let i = 0; i < seeds.length; i++) for (let j = i + 1; j < seeds.length; j++) {
    within.push({ comparison: 'within_learner_different_seeds', province, pair: learner, model_a: learner, model_b: learner,
      seed_a: seeds[i], seed_b: seeds[j], jaccard: jaccard(`${learner}_10km_s${seeds[i]}`, `${learner}_10km_s${seeds[j]}`, provinceValue) });
  }
}

// Reconcile recomputed between-learner values against the saved per-province, per-seed table.
let maxCsvDifference = 0;
for (const row of cross) {
  const saved = csvLookup.get(`${row.model_a}_10km|${row.model_b}_10km|${row.seed_a}|${row.province}`);
  maxCsvDifference = Math.max(maxCsvDifference, Math.abs(row.jaccard - saved));
}
if (maxCsvDifference > 1e-9) throw new Error(`Recomputed Jaccard differs from the saved table by ${maxCsvDifference}.`);

const allRows = [...cross, ...within].sort((a, b) => a.province.localeCompare(b.province) || a.comparison.localeCompare(b.comparison) || a.pair.localeCompare(b.pair) || a.seed_a - b.seed_a || a.seed_b - b.seed_b);
const csvHeader = ['comparison', 'province', 'pair', 'model_a', 'model_b', 'seed_a', 'seed_b', 'jaccard_top10'];
const csvText = [csvHeader.join(','), ...allRows.map(x => csvHeader.map(k => csvEscape(k === 'jaccard_top10' ? x.jaccard : x[k === 'jaccard_top10' ? 'jaccard' : k])).join(','))].join('\n') + '\n';
await writeFile(outCsv, csvText, 'utf8');

const pairSummary = pairs.map(([a, b]) => {
  const between = cross.filter(x => x.pair === `${a}-${b}`).map(x => x.jaccard);
  const noise = within.filter(x => x.pair === a || x.pair === b).map(x => x.jaccard);
  const byProvince = csvProvNames.map(province => {
    const bm = median(cross.filter(x => x.province === province && x.pair === `${a}-${b}`).map(x => x.jaccard));
    const wm = median(within.filter(x => x.province === province && (x.pair === a || x.pair === b)).map(x => x.jaccard));
    return { province, betweenMedian: bm, withinMedian: wm, lower: bm < wm };
  });
  return { pair: `${a}-${b}`, between, noise, byProvince };
});

const byProvinceRows = csvProvNames.map(province => {
  const b = cross.filter(x => x.province === province).map(x => x.jaccard);
  const w = within.filter(x => x.province === province).map(x => x.jaccard);
  const pairsLower = pairSummary.filter(x => x.byProvince.find(y => y.province === province).lower).length;
  return { province, crossMedian: median(b), crossMin: Math.min(...b), crossMax: Math.max(...b),
    withinMedian: median(w), withinMin: Math.min(...w), withinMax: Math.max(...w), pairsLower };
});
const allBetween = cross.map(x => x.jaccard), allWithin = within.map(x => x.jaccard);
const pairRows = pairSummary.map(x => {
  const bm = median(x.between), wm = median(x.noise);
  const minProvinceWins = x.byProvince.filter(y => y.lower).length;
  return `| ${x.pair} | ${fmt(bm)} (${fmt(Math.min(...x.between))}–${fmt(Math.max(...x.between))}) | ${fmt(wm)} (${fmt(Math.min(...x.noise))}–${fmt(Math.max(...x.noise))}) | ${minProvinceWins}/9 |`;
});
const provinceRows = byProvinceRows.map(x => `| ${x.province} | ${fmt(x.crossMedian)} (${fmt(x.crossMin)}–${fmt(x.crossMax)}) | ${fmt(x.withinMedian)} (${fmt(x.withinMin)}–${fmt(x.withinMax)}) | ${x.pairsLower}/3 |`);

const markdown = `# Post-hoc: learner differences versus seed-to-seed variation

**Status:** post-hoc descriptive analysis, 2 October 2026. It uses only the saved 10-km prediction arrays and the locked study-wide top-10% definition; no model was fitted or rerun. It was not specified before outcomes were observed. No pixel-level hypothesis test, p-value, or significance claim is made.

## Question and comparison

Does the between-learner difference in top-decile hotspot maps exceed the map variation from changing the training-background seed within the same learner?

- Between learners: RF–XGB, RF–LR, and XGB–LR, paired at the same seed; 5 seed comparisons per pair and province.
- Within learner: all 10 unique pairs among the 5 seeds for each learner; comparisons are at the same 10-km radius. For each learner pair, the seed-variation reference pools the within-seed comparisons from both learners in that pair.
- Metric: province-specific Jaccard overlap of study-wide top-10% cells, using the same thresholding, map grid, province labels, and NoData mask as the locked comparison. Each province-level summary is first computed independently; no pixel is treated as an independent replicate.

The province-code ordering was recovered by matching recomputed cross-learner Jaccard values against the saved per-province/per-seed table. The recomputed 135 between-learner values reconcile to that table with maximum absolute difference **${maxCsvDifference.toExponential(1)}**.

## Results

Across all 9 provinces, between-learner comparisons have median Jaccard **${fmt(median(allBetween))}** (range ${fmt(Math.min(...allBetween))}–${fmt(Math.max(...allBetween))}); same-learner, different-seed comparisons have median **${fmt(median(allWithin))}** (range ${fmt(Math.min(...allWithin))}–${fmt(Math.max(...allWithin))}). Higher Jaccard means greater map overlap.

| Learner pair | Between-model Jaccard median (range) | Within-model seed Jaccard median (range; both learners pooled) | Provinces where between-model median is lower |
|---|---:|---:|---:|
${pairRows.join('\n')}

| Province | Between-learner Jaccard median (range across pair × seed) | Within-learner seed Jaccard median (range across learner × seed-pair) | Learner pairs with lower between-model median |
|---|---:|---:|---:|
${provinceRows.join('\n')}

**Descriptive criterion:** among the 27 province × learner-pair comparisons, the between-model median is below the corresponding two learners' seed-variation median in **${pairSummary.reduce((sum, p) => sum + p.byProvince.filter(x => x.lower).length, 0)}/27** cases. This indicates whether the observed gap is consistently larger than seed variation under this specific Jaccard summary; it is not a statistical significance result. The full pair-level values are in \`posthoc_jaccard_seed_vs_learner_2026-10-02.csv\`.

## Interpretation limits

- This is a post-hoc robustness comparison, not part of the locked confirmatory analysis. The median/range summaries and the “between median < within median” criterion were chosen after inspecting the run outputs.
- The 5 seeds are five background draws, not independent study datasets. All pairwise seed comparisons reuse the same five fits and are dependent.
- Jaccard captures overlap of the top-decile hotspot set, not similarity of every continuous suitability value. SSIM is not recalculated in this post-hoc comparison.
- Province-level results are shown separately because agreement varies by province. Avoid calling the maps “significantly different”; use wording such as “between-learner hotspot overlap was generally lower than within-learner overlap across seed draws” only if the table supports it.
- No change was made to the pre-declaration, locked outputs, or main v2 readout.
`;
await writeFile(outMd, markdown, 'utf8');

console.log(JSON.stringify({
  csv: path.relative(root, outCsv), markdown: path.relative(root, outMd),
  cross_comparisons: cross.length, within_seed_comparisons: within.length,
  all_between_median: median(allBetween), all_within_median: median(allWithin),
  province_pair_cases_between_lower: pairSummary.reduce((sum, p) => sum + p.byProvince.filter(x => x.lower).length, 0),
  province_pair_cases_total: 27, max_reconciliation_difference: maxCsvDifference,
}, null, 2));
