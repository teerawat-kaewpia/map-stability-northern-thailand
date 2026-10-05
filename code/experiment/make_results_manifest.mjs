import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readdir, readFile, rename, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, '..');
const outPath = path.join(root, '04_notes', 'results_manifest_2026-10-01.json');
const lockPath = path.join(root, '00_predeclaration', 'input_lock_2026-10-01.json');
const lock = JSON.parse(await readFile(lockPath, 'utf8'));

async function listFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...await listFiles(full));
    else if (entry.isFile()) files.push(full);
  }
  return files;
}

function sha256(file) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(file);
    stream.on('data', chunk => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

const outputRoots = ['02_data/runs', '02_data/maps'];
const outputs = [];
for (const relRoot of outputRoots) {
  const absRoot = path.join(root, relRoot);
  for (const file of await listFiles(absRoot)) {
    const info = await stat(file);
    outputs.push({
      path: path.relative(root, file).replaceAll(path.sep, '/'),
      bytes: info.size,
      sha256: await sha256(file),
    });
  }
}
outputs.sort((a, b) => a.path.localeCompare(b.path));

const tmpFiles = outputs.filter(x => x.path.endsWith('.tmp'));
if (tmpFiles.length) throw new Error(`Refusing to freeze results: ${tmpFiles.length} temporary outputs remain.`);
const lopoJson = outputs.filter(x => /^02_data\/runs\/lopo\/TH\d+_.+_s\d+\.json$/.test(x.path));
const predictionMaps = outputs.filter(x => /^02_data\/maps\/(RF|XGB|LR)_.+_s\d+\.npy$/.test(x.path));
if (lopoJson.length !== 270) throw new Error(`Expected 270 LOPO unit JSON files; found ${lopoJson.length}.`);
if (predictionMaps.length !== 90) throw new Error(`Expected 90 prediction maps; found ${predictionMaps.length}.`);

const provenanceFiles = [
  '00_predeclaration/predeclaration_2026-10-01.md',
  '00_predeclaration/input_lock_2026-10-01.json',
  '01_scripts/run_map_stability.py',
  '01_scripts/make_results_manifest.mjs',
  '01_scripts/make_results_readout.mjs',
  '01_scripts/make_test_points.py',
  '01_scripts/make_seed_vs_learner_posthoc.mjs',
  '04_notes/results_readout_2026-10-01.md',
  '04_notes/results_readout_2026-10-01_v2.md',
  '04_notes/posthoc_jaccard_seed_vs_learner_2026-10-02.csv',
  '04_notes/posthoc_seed_vs_learner_2026-10-02.md',
  '04_notes/run_log_2026-10-01.txt',
  '04_notes/run_log_2026-10-01_resume1.txt',
];
const provenance = [];
for (const rel of provenanceFiles) {
  const full = path.join(root, rel);
  provenance.push({ path: rel, bytes: (await stat(full)).size, sha256: await sha256(full) });
}
if (lock.runner?.sha256 !== provenance.find(x => x.path.endsWith('run_map_stability.py'))?.sha256) {
  throw new Error('Runner hash does not match the recorded input lock.');
}
if (lock.predeclaration?.sha256 !== provenance.find(x => x.path.endsWith('predeclaration_2026-10-01.md'))?.sha256) {
  throw new Error('Pre-declaration hash does not match the recorded input lock.');
}

const manifest = {
  schema: 'exp_map_stability_results_manifest_v1',
  created_at_utc: new Date().toISOString(),
  run_completed_at_reported: '2026-10-01 22:22:58 Asia/Bangkok',
  scope: outputRoots,
  output_file_count: outputs.length,
  lopo_unit_json_count: lopoJson.length,
  prediction_map_count: predictionMaps.length,
  provenance_files: provenance,
  outputs,
};
const tmpPath = `${outPath}.tmp`;
await writeFile(tmpPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
await rename(tmpPath, outPath);
console.log(JSON.stringify({ path: path.relative(root, outPath), output_file_count: outputs.length,
  lopo_unit_json_count: lopoJson.length, prediction_map_count: predictionMaps.length,
  predeclaration_hash: provenance.find(x => x.path.endsWith('predeclaration_2026-10-01.md')).sha256,
  runner_hash: provenance.find(x => x.path.endsWith('run_map_stability.py')).sha256 }, null, 2));
