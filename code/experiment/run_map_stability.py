"""
Map-stability experiment - runner.

Implements predeclaration_2026-10-01.md (original text + Amendments 1-3) and the runner-level choices
listed in RUNNER_CHOICES below, which the predeclaration does not fix and which must be locked as
Amendment 4 before any fit.

Modes
  --dry-run : verifies every locked hash, builds the 500 m map grid and mask, counts inner spatial
              blocks per fold from presences only. Draws no background and fits nothing.
  --run     : full experiment. REFUSES to start unless the input lock (a) records an Amendment with
              id 4 and (b) records this file's SHA-256 under "runner". Resumable: every unit writes its
              own output and is skipped if that output already exists.

Outputs only under exp_map_stability/02_data/runs, 02_data/maps and 04_notes.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, pyproj, rasterio
from rasterio.transform import rowcol, from_origin
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter
from scipy.stats import rankdata, spearmanr
from shapely.ops import unary_union
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
EXP  = Path(__file__).resolve().parents[1]
LOCK = EXP / "00_predeclaration" / "input_lock_2026-10-01.json"
RUNS, MAPS, NOTES = EXP / "02_data" / "runs", EXP / "02_data" / "maps", EXP / "04_notes"

# ---------------------------------------------------------------- locked by the predeclaration
RADII    = [("1km", 1000.0), ("5km", 5000.0), ("10km", 10000.0), ("20km", 20000.0),
            ("25km", 25000.0), ("Rinf", None)]                         # design table
LEARNERS = ["RF", "XGB", "LR"]
SEEDS    = [42, 0, 1, 7, 2024]                                         # training-bg order only (A1.5)
INNER_M  = 500.0
MODEL_RS = 42                                                          # A1.5
REF_RADIUS = "10km"                                                    # similarity reference
SSIM_WINDOWS = [7, 21]                                                 # primary, sensitivity (A1.7)
FACTOR_GROUPS = {                                                      # supplementary LOFO groups
    "development_intensity": ["builtup", "nightlight"],
    "transport_access": ["dist_road_km", "dist_rail_km", "dist_railway_new_km"],
    "market_gateway_access": ["dist_cbd_km", "dist_dryport_km", "dist_border_CK_km"],
    "physical_capacity": ["slope_deg", "dist_water_km"],
    "forest_area_condition": ["forest"]}
PERM_GROUP = ["builtup", "nightlight"]                                 # correlated pair, tested as group
SHAP_PERM_SEED, SHAP_PERM_MAX_EVALS = 161803, 115                      # A4.10: LR, 5 x (2 x 11 + 1)

# ---------------------------------------------------------------- RUNNER_CHOICES (-> Amendment 4)
RUNNER_CHOICES = {
 "U1_map_grid": "500 m grid built from the LOCKED boundary (union of the 9 polygons), origin = bounds "
                "floored/ceiled to 500 m; mask = rasterized boundary AND all 11 covariates finite; "
                "covariates warped from the locked rasters, bilinear except forest/builtup (nearest); "
                "5 distance rasters m->km; dist_border_CK_km analytic at cell centres (A1.8). "
                "Differs from Paper C's published map grid, which used GADM 4.1.",
 "U2_spatial_blocks": "absolute 50 km grid: block = (floor(E/50000), floor(N/50000)), as in Paper C's "
                      "LOPO spatial_block_auc; NOT paperA_validation_utils.spatial_block_labels, whose "
                      "origin is the data minimum and would move with each background draw.",
 "U3_threshold_oof": "each learner gets its own leave-one-block-out OOF scores on its fold's training "
                     "data (training presences + that unit's training background); a block is skipped "
                     "if the remaining training data lack a class; thresholds = distinct finite OOF "
                     "scores (roc_curve, drop_intermediate=False), max TSS, ties -> highest.",
 "U4_maps": "map models fitted on all nine provinces (A3.1, n_bg = 2,859) for every radius x learner x "
            "seed (90 maps); similarity computed within a seed and summarised as median, min and max "
            "across the five seeds (map_similarity_summary_across_seeds.csv).",
 "U5_ssim_nodata": "skimage structural_similarity(full=True, win_size=w, data_range=1, uniform "
                   "window); NoData filled with 0 only for the computation; the SSIM map is averaged "
                   "over cells of the province whose entire w x w window is valid (NoData-touching "
                   "windows and array edges excluded).",
 "U6_rank_maps": "percentile rank over all valid study-area cells: rankdata(method='average') / n.",
 "U7_top10": "top 10% = score >= the 90th percentile of all valid study-area cells (study-wide, per map); "
             "Jaccard computed on each province's cells, reported with the hotspot counts behind it "
             "(n in a, n in b, intersection, union); empty union -> NaN.",
 "U8_permutation": "on each fold's shared test set; drop in AUC; 10 repeats; fresh "
                   "numpy default_rng(42) per feature/group; the group builtup+nightlight permuted with "
                   "one shared row permutation; all folds, radii, learners and seeds.",
 "U9_shap": "seed-42 map models only (18); explained cells = 1,000 valid cells drawn with "
            "default_rng(314159); reference background = 100 valid cells drawn with "
            "default_rng(271828); output = P(presence) for all learners: RF/XGB TreeExplainer "
            "(interventional, model_output='probability'), LR shap PermutationExplainer on "
            "predict_proba[:,1] with an Independent masker on the same 100 cells, seed 161803, "
            "max_evals 115; summary rebuilt from every valid SHAP file on disk.",
 "U10_lofo": "seed 42 only - conclusions conditional on that background draw; every fold x radius x "
             "learner x the 5 pre-declared groups; refit without the group; AUC drop on the fold's "
             "shared test set; no threshold.",
 "U11_metrics": "TSS = sensitivity + specificity - 1 at the frozen threshold (positive if score >= t); "
                "Kappa = Cohen's kappa at the same threshold; fold spread reported as min, IQR, max.",
 "U12_no_imputation": "no imputer: all training and test rows and all mapped cells must have finite "
                      "covariates (the runner asserts this) - Paper C's median imputer is not needed.",
 "U13_presence_covariates": "presence covariates are taken as stored in the locked clean presence file.",
 "U14_resume": "every output is written to a .tmp file and renamed when complete; on resume a unit is "
               "skipped only if its files load and pass shape/content checks (LOPO JSON written last "
               "as the commit marker; maps: length = valid cells, finite, in [0,1]; seed-42 models load).",
}

FEATURES = ["slope_deg", "dist_road_km", "dist_rail_km", "dist_water_km", "dist_cbd_km",
            "dist_dryport_km", "nightlight", "forest", "builtup", "dist_railway_new_km",
            "dist_border_CK_km"]
LAYERS = [("slope_deg", "grid_slope_17prov.tif", 1.0), ("dist_road_km", "grid_dist_road_17prov.tif", 1e-3),
          ("dist_rail_km", "grid_dist_rail_17prov.tif", 1e-3), ("dist_water_km", "grid_dist_water_17prov.tif", 1e-3),
          ("dist_cbd_km", "grid_dist_cbd_17prov.tif", 1e-3), ("dist_dryport_km", "grid_dist_dryport_17prov.tif", 1e-3),
          ("nightlight", "grid_nl_17prov.tif", 1.0), ("forest", "grid_forest_2568_17prov.tif", 1.0),
          ("builtup", "grid_builtup_17prov.tif", 1.0), ("dist_railway_new_km", "dist_railway_new_km.tif", 1.0)]
NEAREST = {"forest", "builtup"}
CK_E, CK_N = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True).transform(100.0894, 20.3622)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---- U14: atomic writes + validation before a unit is skipped on resume ----------------------
def _commit(tmp, final): os.replace(tmp, final)


def save_npy(path, arr):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    _commit(tmp, path)


def save_text(path, text):
    tmp = Path(str(path) + ".tmp"); tmp.write_text(text, encoding="utf-8"); _commit(tmp, path)


def save_csv(path, df):
    tmp = Path(str(path) + ".tmp"); df.to_csv(tmp, index=False, encoding="utf-8-sig"); _commit(tmp, path)


def save_joblib(path, obj):
    import joblib
    tmp = Path(str(path) + ".tmp"); joblib.dump(obj, tmp); _commit(tmp, path)


def ok_npy(path, shape, prob=False):
    """True only if the file loads, has the expected shape, is finite and (if prob) lies in [0, 1]."""
    try:
        a = np.load(path)
    except Exception:
        return False
    if a.shape != tuple(shape) or not np.isfinite(a).all():
        return False
    return not prob or (a.min() >= 0 and a.max() <= 1)


def ok_json(path, required=()):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return d if all(k in d for k in required) else None


def ok_joblib(path):
    import joblib
    try:
        joblib.load(path); return True
    except Exception:
        return False


# ================================================================= 0. verification
def verify(lock, mode):
    for rec in lock["inputs"]:
        if sha(rec["file"]) != rec["sha256"]:
            raise SystemExit(f"HASH MISMATCH, abort: {rec['file']}")
    root = EXP.parent                      # RF_RP: lock paths start at exp_map_stability/
    for rec in lock["test_points"]["files"] + [lock["test_points"]["manifest"]]:
        if sha(root / rec["file"]) != rec["sha256"]:
            raise SystemExit(f"TEST-POINT HASH MISMATCH, abort: {rec['file']}")
    if sha(EXP / "00_predeclaration" / "predeclaration_2026-10-01.md") != lock["predeclaration"]["sha256"]:
        raise SystemExit("predeclaration hash differs from lock, abort")
    if mode == "run":
        if not any(a.get("id") == 4 for a in lock.get("amendments", [])):
            raise SystemExit("REFUSED: no Amendment 4 in the lock (runner choices not locked)")
        if lock.get("runner", {}).get("sha256") != sha(__file__):
            raise SystemExit("REFUSED: this runner's SHA-256 is not the one recorded in the lock")
    log(f"verified {len(lock['inputs'])} inputs, {len(lock['test_points']['files'])} test-point files, predeclaration")


# ================================================================= 1. data
class Data:
    def __init__(self, lock):
        p = {Path(i["file"]).name: i["file"] for i in lock["inputs"]}
        self.pres = pd.read_csv(p["analysis_dataset_m3_base_clean.csv"], encoding="utf-8-sig")
        assert len(self.pres) == 2859 and (self.pres["presence"] == 1).all()
        assert np.isfinite(self.pres[FEATURES].to_numpy()).all()                       # U12
        self.bnd = gpd.read_file(p["boundary_9prov_32647.gpkg"]).to_crs("EPSG:32647").sort_values("ADM1_PCODE")
        self.poly = dict(zip(self.bnd["ADM1_TH"], self.bnd.geometry))
        self.code = dict(zip(self.bnd["ADM1_TH"], self.bnd["ADM1_PCODE"]))
        self.provs = list(self.bnd["ADM1_TH"])
        self.raster_paths = {f: p[fn] for f, fn, _ in LAYERS}
        root = EXP.parent                      # RF_RP: lock paths start at exp_map_stability/
        self.test_bg = {r["ADM1_PCODE"]: pd.read_csv(root / r["file"], encoding="utf-8-sig")
                        for r in lock["test_points"]["files"]}
        self._r = None

    def rasters(self):                                    # read each locked raster once
        if self._r is None:
            self._r = {}
            for f, fn, fac in LAYERS:
                with rasterio.open(self.raster_paths[f]) as s:
                    a = s.read(1).astype("float32"); nod = s.nodata if s.nodata is not None else -9999
                    a[a == nod] = np.nan
                    self._r[f] = (a * fac if fac != 1.0 else a, s.transform)
        return self._r

    def extract(self, e, n):
        out = {}
        for f, (a, tr) in self.rasters().items():
            r, c = rowcol(tr, e, n); r, c = np.asarray(r), np.asarray(c)
            ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
            v = np.full(len(e), np.nan, "float64"); v[ok] = a[r[ok], c[ok]]
            out[f] = v
        out["dist_border_CK_km"] = np.hypot(e - CK_E, n - CK_N) / 1000.0
        return pd.DataFrame(out)


# ================================================================= 2. background (A3.2-A3.4)
def sample_bg(D, train_xy, polygon, n_bg, outer_m, seed):
    rng = np.random.default_rng(seed)
    tree = cKDTree(train_xy)
    minx, miny, maxx, maxy = polygon.bounds
    kept, got, rounds = [], 0, 0
    for i in range(1, 7):
        rounds = i
        k = i * max(20 * n_bg, 20000)
        e = rng.uniform(minx, maxx, k)
        n = rng.uniform(miny, maxy, k)
        inside = gpd.GeoSeries(gpd.points_from_xy(e, n), crs="EPSG:32647").within(polygon).to_numpy()
        d, _ = tree.query(np.column_stack([e, n]), k=1)
        keep = inside & (d >= INNER_M)
        if outer_m is not None:                                   # A3.4: R-inf drops only this
            keep &= d <= outer_m
        if keep.any():
            kept.append(np.column_stack([e[keep], n[keep]])); got += int(keep.sum())
        if got >= 3 * n_bg:
            break
    info = {"rounds": rounds, "kept_geom": got}
    if not kept:
        return None, info
    cand = np.vstack(kept)
    feat = D.extract(cand[:, 0], cand[:, 1])
    feat["easting"], feat["northing"] = cand[:, 0], cand[:, 1]
    feat = feat[np.isfinite(feat[FEATURES].to_numpy()).all(axis=1)]
    info["kept_complete"] = int(len(feat))
    if len(feat) < n_bg:
        return None, info
    return feat.head(n_bg).reset_index(drop=True), info


# ================================================================= 3. learners, threshold, metrics
def make(learner):
    if learner == "RF":
        return RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=5,
                                      class_weight="balanced", random_state=MODEL_RS, n_jobs=-1)
    if learner == "XGB":
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=MODEL_RS,
                             n_jobs=-1, verbosity=0)
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000,
                                               class_weight="balanced"))])


def blocks(df):                                                       # U2
    return (np.floor(df["easting"].to_numpy() / 50000).astype(int).astype(str) + "_" +
            np.floor(df["northing"].to_numpy() / 50000).astype(int).astype(str))


def threshold_from_oof(learner, train, feats=FEATURES):               # A1.6 + U3
    b = blocks(train); y = train["presence"].to_numpy(); X = train[feats]
    oof = np.full(len(train), np.nan)
    for blk in np.unique(b):
        te = b == blk; tr = ~te
        if y[tr].min() == y[tr].max():
            continue
        oof[te] = make(learner).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    ok = np.isfinite(oof)
    t, best = select_threshold(y[ok], oof[ok])
    return t, best, int(ok.sum()), int(len(np.unique(b)))


TIE_TOL = 1e-12


def select_threshold(y, s):
    """Max TSS over distinct finite scores; positive if s >= t; ties -> highest finite threshold.
    TSS values that are mathematically equal can differ in the last floating-point digits, so a tie
    is any TSS within TIE_TOL of the maximum (a plain argmax would pick an arbitrary member)."""
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    fin = np.isfinite(thr)
    tss, thr = (tpr - fpr)[fin], thr[fin]
    i = int(np.flatnonzero(tss >= tss.max() - TIE_TOL)[0])   # thresholds decrease -> first = highest
    return float(thr[i]), float(tss[i])


def tss_kappa(y, s, t):
    p = (s >= t).astype(int)
    tp = ((p == 1) & (y == 1)).sum(); fn = ((p == 0) & (y == 1)).sum()
    tn = ((p == 0) & (y == 0)).sum(); fp = ((p == 1) & (y == 0)).sum()
    return float(tp / (tp + fn) + tn / (tn + fp) - 1), float(cohen_kappa_score(y, p))


def permutation_drop(model, X, y, cols_list, base):                   # U8
    out = {}
    for cols in cols_list:
        rng = np.random.default_rng(42); drops = []
        for _ in range(10):
            Xp = X.copy(); idx = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[idx]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out["+".join(cols)] = (float(np.mean(drops)), float(np.std(drops)))
    return out


# ================================================================= 4. LOPO
def lopo_unit_done(f, fbg, n_bg, seed):
    """A unit is skipped only if its JSON parses and is internally complete, and - for a complete
    unit - its background file loads with the expected shape."""
    d = ok_json(f, ("status", "fold", "radius", "seed"))
    if d is None:
        return False
    if d["status"].startswith("incomplete"):
        return True
    L = d.get("learners", {})
    need = {"auc", "tss", "kappa", "threshold", "perm_auc_drop"} | ({"lofo_auc_drop"} if seed == 42 else set())
    if set(L) != set(LEARNERS) or any(not need <= set(L[k]) for k in LEARNERS):
        return False
    return ok_npy(fbg, (n_bg, 2))


def run_lopo(D):
    out = RUNS / "lopo"; out.mkdir(parents=True, exist_ok=True)
    for held in D.provs:
        code = D.code[held]
        pres_tr = D.pres[D.pres["FPROVNAME"] != held]; pres_te = D.pres[D.pres["FPROVNAME"] == held]
        poly_tr = unary_union([D.poly[p] for p in D.provs if p != held])
        tb = D.test_bg[code]
        test = pd.concat([pres_te[["easting", "northing"] + FEATURES].assign(presence=1),
                          tb[["easting", "northing"] + FEATURES].assign(presence=0)], ignore_index=True)
        assert np.isfinite(test[FEATURES].to_numpy()).all()
        n_bg = len(pres_tr)
        for seed in SEEDS:
            for rlab, outer in RADII:
                f = out / f"{code}_{rlab}_s{seed}.json"
                fbg = out / f"{code}_{rlab}_s{seed}_bg.npy"
                if lopo_unit_done(f, fbg, n_bg, seed):
                    continue
                t0 = time.time()
                bg, info = sample_bg(D, pres_tr[["easting", "northing"]].to_numpy(), poly_tr, n_bg, outer, seed)
                rec = {"fold": code, "held_out": held, "radius": rlab, "seed": seed, "n_bg": n_bg,
                       "n_test_presence": len(pres_te), "n_test_bg": len(tb), "sampling": info}
                if bg is None:
                    rec["status"] = "incomplete under the locked training-sampling budget"
                    save_text(f, json.dumps(rec, ensure_ascii=False, indent=1)); continue
                train = pd.concat([pres_tr[["easting", "northing"] + FEATURES].assign(presence=1),
                                   bg[["easting", "northing"] + FEATURES].assign(presence=0)], ignore_index=True)
                save_npy(fbg, bg[["easting", "northing"]].to_numpy())
                rec["status"], rec["learners"] = "complete", {}
                for L in LEARNERS:
                    thr, tss_in, n_oof, n_blk = threshold_from_oof(L, train)
                    m = make(L).fit(train[FEATURES], train["presence"])
                    s = m.predict_proba(test[FEATURES])[:, 1]; y = test["presence"].to_numpy()
                    auc = float(roc_auc_score(y, s)); tss, kap = tss_kappa(y, s, thr)
                    perm = permutation_drop(m, test[FEATURES], y, [[c] for c in FEATURES] + [PERM_GROUP], auc)
                    lr = {"auc": auc, "tss": tss, "kappa": kap, "threshold": thr, "threshold_train_tss": tss_in,
                          "n_oof_rows": n_oof, "n_inner_blocks": n_blk, "perm_auc_drop": perm}
                    if seed == 42:                                                   # U10
                        lr["lofo_auc_drop"] = {}
                        for g, cols in FACTOR_GROUPS.items():
                            keep = [c for c in FEATURES if c not in cols]
                            mg = make(L).fit(train[keep], train["presence"])
                            lr["lofo_auc_drop"][g] = auc - float(roc_auc_score(y, mg.predict_proba(test[keep])[:, 1]))
                    rec["learners"][L] = lr
                rec["seconds"] = round(time.time() - t0, 1)
                save_text(f, json.dumps(rec, ensure_ascii=False, indent=1))     # written last = commit
                log(f"LOPO {code} {rlab:<5} s{seed:<4} " +
                    " ".join(f"{L}:AUC={rec['learners'][L]['auc']:.3f}" for L in LEARNERS) + f" ({rec['seconds']}s)")


# ================================================================= 5. grid + maps
def build_grid(D):                                                    # U1
    union = unary_union(list(D.bnd.geometry)); RES = 500.0
    minx, miny, maxx, maxy = union.bounds
    minx, miny = np.floor(minx / RES) * RES, np.floor(miny / RES) * RES
    maxx, maxy = np.ceil(maxx / RES) * RES, np.ceil(maxy / RES) * RES
    W, H = int((maxx - minx) / RES), int((maxy - miny) / RES)
    tr = from_origin(minx, maxy, RES, RES)
    mask = rasterize([(g, 1) for g in D.bnd.geometry], out_shape=(H, W), transform=tr, fill=0, dtype="uint8").astype(bool)
    prov = rasterize([(g, i + 1) for i, g in enumerate(D.bnd.geometry)], out_shape=(H, W), transform=tr,
                     fill=0, dtype="uint8")
    layers = {}
    for f, fn, fac in LAYERS:
        with rasterio.open(D.raster_paths[f]) as s:
            dst = np.full((H, W), np.nan, "float32")
            reproject(rasterio.band(s, 1), dst, src_transform=s.transform, src_crs=s.crs, dst_transform=tr,
                      dst_crs="EPSG:32647", resampling=Resampling.nearest if f in NEAREST else Resampling.bilinear,
                      src_nodata=s.nodata, dst_nodata=np.nan)
        layers[f] = dst * fac if fac != 1.0 else dst
    xs = minx + (np.arange(W) + 0.5) * RES; ys = maxy - (np.arange(H) + 0.5) * RES
    layers["dist_border_CK_km"] = (np.hypot(xs[None, :] - CK_E, ys[:, None] - CK_N) / 1000.0).astype("float32")
    valid = mask & np.all([np.isfinite(layers[f]) for f in FEATURES], axis=0)
    X = pd.DataFrame({f: layers[f][valid] for f in FEATURES})
    return {"H": H, "W": W, "transform": tr, "valid": valid, "prov": prov, "X": X,
            "prov_codes": list(D.bnd["ADM1_PCODE"])}


def run_maps(D, G):
    MAPS.mkdir(parents=True, exist_ok=True)
    save_npy(MAPS / "valid_mask.npy", G["valid"]); save_npy(MAPS / "province_raster.npy", G["prov"])
    nv = len(G["X"])
    all_xy = D.pres[["easting", "northing"]].to_numpy()
    poly9 = unary_union(list(D.bnd.geometry))
    for seed in SEEDS:
        for rlab, outer in RADII:
            inc = MAPS / f"INCOMPLETE_{rlab}_s{seed}.json"
            maps_ok = all(ok_npy(MAPS / f"{L}_{rlab}_s{seed}.npy", (nv,), prob=True) for L in LEARNERS)
            models_ok = seed != 42 or all(ok_joblib(MAPS / f"model_{L}_{rlab}_s42.joblib") for L in LEARNERS)
            if (maps_ok and models_ok) or ok_json(inc, ("rounds",)) is not None:
                continue
            bg, info = sample_bg(D, all_xy, poly9, 2859, outer, seed)
            if bg is None:
                save_text(inc, json.dumps(info)); continue
            train = pd.concat([D.pres[FEATURES].assign(presence=1), bg[FEATURES].assign(presence=0)], ignore_index=True)
            for L in LEARNERS:
                m = make(L).fit(train[FEATURES], train["presence"])
                if seed == 42:                                        # model before map: map = commit
                    save_joblib(MAPS / f"model_{L}_{rlab}_s42.joblib", m)
                save_npy(MAPS / f"{L}_{rlab}_s{seed}.npy", m.predict_proba(G["X"][FEATURES])[:, 1].astype("float32"))
            log(f"MAP {rlab:<5} s{seed:<4} done")


# ================================================================= 6. similarity
def to_grid(G, v):
    a = np.full((G["H"], G["W"]), np.nan, "float32"); a[G["valid"]] = v; return a


def ssim_by_province(G, a, b, w):                                     # U5
    from skimage.metrics import structural_similarity
    _, S = structural_similarity(np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0), win_size=w,
                                 data_range=1.0, full=True)
    winvalid = uniform_filter(G["valid"].astype("float64"), size=w, mode="constant", cval=0.0) >= 1 - 1e-9
    return {c: float(S[(G["prov"] == i + 1) & winvalid].mean()) for i, c in enumerate(G["prov_codes"])}


def jaccard_by_province(G, va, vb):                                   # U7
    """Study-wide top 10% of each map; per province: Jaccard plus the hotspot counts behind it."""
    ta = to_grid(G, (va >= np.quantile(va, 0.9)).astype("float32")) == 1
    tb = to_grid(G, (vb >= np.quantile(vb, 0.9)).astype("float32")) == 1
    out, counts = {}, {}
    for i, c in enumerate(G["prov_codes"]):
        pm = G["prov"] == i + 1
        na, nb = int((ta & pm).sum()), int((tb & pm).sum())
        ni, nu = int((ta & tb & pm).sum()), int(((ta | tb) & pm).sum())
        out[c] = float(ni / nu) if nu else float("nan")
        counts[c] = {"n_top10_a": na, "n_top10_b": nb, "n_intersection": ni, "n_union": nu}
    return out, counts


def run_similarity(G):
    rows = []
    nv = int(G["valid"].sum())
    load = lambda L, r, s: (np.load(MAPS / f"{L}_{r}_s{s}.npy")
                            if ok_npy(MAPS / f"{L}_{r}_s{s}.npy", (nv,), prob=True) else None)
    pairs = [("across_radii", L, (L, r), (L, REF_RADIUS)) for L in LEARNERS for r, _ in RADII if r != REF_RADIUS] + \
            [("across_learners", r, (a, r), (b, r)) for r, _ in RADII for a, b in [("RF", "XGB"), ("RF", "LR"), ("XGB", "LR")]]
    for seed in SEEDS:
        for kind, within, (La, ra), (Lb, rb) in pairs:
            va, vb = load(La, ra, seed), load(Lb, rb, seed)
            base = {"seed": seed, "comparison": kind, "within": within, "a": f"{La}_{ra}", "b": f"{Lb}_{rb}"}
            if va is None or vb is None:
                rows.append({**base, "status": "map missing (incomplete sampling)"}); continue
            ra_, rb_ = rankdata(va, method="average") / len(va), rankdata(vb, method="average") / len(vb)  # U6
            res = {}
            for w in SSIM_WINDOWS:
                res[f"ssim_raw_{w}"] = ssim_by_province(G, to_grid(G, va), to_grid(G, vb), w)
                res[f"ssim_rank_{w}"] = ssim_by_province(G, to_grid(G, ra_), to_grid(G, rb_), w)
            res["jaccard_top10"], counts = jaccard_by_province(G, va, vb)
            for metric, byprov in res.items():
                for code, val in byprov.items():
                    extra = counts[code] if metric == "jaccard_top10" else {}
                    rows.append({**base, "status": "ok", "metric": metric, "province": code, "value": val, **extra})
            if seed == 42:                                            # difference map of the screening area
                ta, tb = va >= np.quantile(va, 0.9), vb >= np.quantile(vb, 0.9)
                cls = np.where(ta & tb, 2, np.where(ta, 1, np.where(tb, -1, 0))).astype("int8")
                save_npy(MAPS / f"diff_top10_{La}_{ra}_vs_{Lb}_{rb}_s42.npy", cls)
    sim = pd.DataFrame(rows)
    save_csv(RUNS / "map_similarity_by_province.csv", sim)
    log(f"similarity rows: {len(rows)}")
    # A4.5: within-seed values summarised across the five seeds
    ok = sim[sim["status"] == "ok"]
    key = ["comparison", "within", "a", "b", "metric", "province"]
    summ = ok.groupby(key)["value"].agg(median="median", min="min", max="max",
                                        n_seeds_with_value="count").reset_index()
    summ["n_seeds_compared"] = ok.groupby(key)["seed"].nunique().values
    save_csv(RUNS / "map_similarity_summary_across_seeds.csv", summ)


def summarize_lopo():                                                 # A4.12 (+ A4.5 for seeds)
    recs = [ok_json(p, ("status",)) for p in sorted((RUNS / "lopo").glob("*.json"))]
    rows, status = [], []
    for d in filter(None, recs):
        status.append({k: d[k] for k in ("fold", "radius", "seed", "status")})
        for L, v in d.get("learners", {}).items():
            rows.append({"fold": d["fold"], "radius": d["radius"], "seed": d["seed"], "learner": L,
                         "auc": v["auc"], "tss": v["tss"], "kappa": v["kappa"], "threshold": v["threshold"]})
    save_csv(RUNS / "lopo_unit_status.csv", pd.DataFrame(status))
    if not rows:
        return
    df = pd.DataFrame(rows); save_csv(RUNS / "lopo_metrics_by_fold.csv", df)
    q = lambda p: (lambda s: s.quantile(p))
    by_seed = (df.groupby(["learner", "radius", "seed"])[["auc", "tss", "kappa"]]
                 .agg(["median", "min", q(0.25), q(0.75), "max", "count"]))
    by_seed.columns = ["_".join(c).replace("<lambda_0>", "q25").replace("<lambda_1>", "q75") for c in by_seed.columns]
    by_seed = by_seed.reset_index(); save_csv(RUNS / "lopo_fold_spread_by_seed.csv", by_seed)
    across = (by_seed.groupby(["learner", "radius"])[["auc_median", "tss_median", "kappa_median"]]
                     .agg(["median", "min", "max", "count"]))
    across.columns = ["_".join(c) for c in across.columns]
    save_csv(RUNS / "lopo_fold_median_across_seeds.csv", across.reset_index())


# ================================================================= 7. SHAP (U9)
def run_shap(G):
    import shap, joblib
    n = len(G["X"])
    cells = np.sort(np.random.default_rng(314159).choice(n, 1000, replace=False))
    bgi = np.sort(np.random.default_rng(271828).choice(n, 100, replace=False))
    Xc, Xb = G["X"].iloc[cells][FEATURES], G["X"].iloc[bgi][FEATURES]
    out = RUNS / "shap"; out.mkdir(parents=True, exist_ok=True)
    save_npy(out / "explained_cells.npy", cells); save_npy(out / "background_cells.npy", bgi)
    shape = (len(cells), len(FEATURES))
    for L in LEARNERS:
        for r, _ in RADII:
            p, fo = MAPS / f"model_{L}_{r}_s42.joblib", out / f"shap_{L}_{r}.npy"
            if not ok_joblib(p) or ok_npy(fo, shape):
                continue
            m = joblib.load(p)
            if L == "LR":                                             # A4.10: sampled permutations
                f = lambda X, m=m: m.predict_proba(pd.DataFrame(X, columns=FEATURES))[:, 1]
                ex = shap.explainers.Permutation(f, shap.maskers.Independent(Xb.to_numpy(), max_samples=100),
                                                 seed=SHAP_PERM_SEED)
                sv = ex(Xc.to_numpy(), max_evals=SHAP_PERM_MAX_EVALS, silent=True).values
            else:
                sv = shap.TreeExplainer(m, data=Xb, feature_perturbation="interventional",
                                        model_output="probability").shap_values(Xc)
                sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv
            save_npy(fo, np.asarray(sv, dtype="float64"))
            log(f"SHAP {L} {r} done")
    # summary rebuilt from every valid SHAP file on disk, not only those computed in this session
    rows, missing = [], []
    for L in LEARNERS:
        for r, _ in RADII:
            fo = out / f"shap_{L}_{r}.npy"
            if not ok_npy(fo, shape):
                missing.append(f"{L}_{r}"); continue
            sv = np.load(fo)
            for j, f in enumerate(FEATURES):
                rows.append({"learner": L, "radius": r, "feature": f, "mean_abs_shap": float(np.abs(sv[:, j]).mean()),
                             "direction_spearman": float(spearmanr(Xc[f], sv[:, j])[0])})
    save_csv(out / "shap_summary.csv", pd.DataFrame(rows))
    save_text(out / "shap_summary_coverage.json",
              json.dumps({"expected": len(LEARNERS) * len(RADII), "present": len(LEARNERS) * len(RADII) - len(missing),
                          "missing": missing}, indent=1))
    log(f"SHAP summary: {len(LEARNERS) * len(RADII) - len(missing)}/{len(LEARNERS) * len(RADII)} models")


# ================================================================= main
def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true"); g.add_argument("--run", action="store_true")
    a = ap.parse_args()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    verify(lock, "run" if a.run else "dry")
    D = Data(lock)
    if a.dry_run:
        G = build_grid(D)
        log(f"grid {G['W']}x{G['H']} @500 m; valid cells {int(G['valid'].sum()):,}; "
            f"mask cells {int((G['prov'] > 0).sum()):,}")
        for held in D.provs:
            tr = D.pres[D.pres["FPROVNAME"] != held]
            log(f"fold {D.code[held]}: training presences {len(tr)}, occupied 50 km blocks {len(np.unique(blocks(tr)))}")
        log(f"units: LOPO {len(D.provs) * len(SEEDS) * len(RADII)} (x3 learners), maps {len(SEEDS) * len(RADII)} (x3)")
        log("dry run: no background drawn, no model fitted")
        return
    NOTES.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
    G = build_grid(D)
    run_lopo(D); summarize_lopo(); run_maps(D, G); run_similarity(G); run_shap(G)
    log("all stages complete")


if __name__ == "__main__":
    main()
