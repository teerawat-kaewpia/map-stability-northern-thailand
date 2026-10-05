"""
Map-stability experiment - generate the nine shared LOPO test-background sets (ONLY this step).

Implements predeclaration A1.4 + A2.2 (+ A1.2 province membership, A1.8 covariate 11).
No model is fitted here. Run once; outputs and their SHA-256 are then recorded in the input lock.

Implementation choices not spelled out in the amendments, fixed here and recorded (by this file's
SHA-256) in the input lock BEFORE the first execution:
  * one fresh generator numpy.random.default_rng(314159) per held-out province, so each province's
    points are independent of the order in which provinces are processed;
  * the whole budget B = max(20000, 200 * n_test) is drawn at once for a province: all B eastings,
    then all B northings, uniform in that province polygon's bounding box; draw index j = (e[j], n[j]);
  * filters, applied to every candidate: inside the province polygon (shapely `within`), nearest of
    all 2,859 clean presences >= 500 m, and no missing value in any of the 11 covariates;
  * the first n_test candidates passing all filters, in draw index order, are the test points;
  * if fewer than n_test pass within B, the province is "incomplete under the locked sampling
    budget": the points found are written with an _INCOMPLETE suffix and the fold is not scored;
  * provinces are processed in ADM1_PCODE order (TH50..TH58); n_test is the FPROVNAME count.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, pyproj, rasterio
from rasterio.transform import rowcol
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
EXP   = Path(__file__).resolve().parents[1]
LOCK  = EXP / "00_predeclaration" / "input_lock_2026-10-01.json"
OUT   = EXP / "02_data" / "test_points"
SEED, INNER_M = 314159, 500.0

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

lock = json.loads(LOCK.read_text(encoding="utf-8"))
locked = {Path(i["file"]).name: i for i in lock["inputs"]}
for name, rec in locked.items():                       # refuse to run on any drifted input
    if sha(rec["file"]) != rec["sha256"]:
        raise SystemExit(f"HASH MISMATCH, abort: {rec['file']}")
path = lambda n: locked[n]["file"]

PRES = pd.read_csv(path("analysis_dataset_m3_base_clean.csv"), encoding="utf-8-sig")
assert len(PRES) == 2859 and (PRES["presence"] == 1).all()
BND  = gpd.read_file(path("boundary_9prov_32647.gpkg")).to_crs("EPSG:32647").sort_values("ADM1_PCODE")
TREE = cKDTree(PRES[["easting", "northing"]].to_numpy())

# covariates: same files as the lock, same unit factors as phase1b_radius_comparison.py
LAYERS = [("slope_deg", "grid_slope_17prov.tif", 1.0),
          ("dist_road_km", "grid_dist_road_17prov.tif", 1/1000),
          ("dist_rail_km", "grid_dist_rail_17prov.tif", 1/1000),
          ("dist_water_km", "grid_dist_water_17prov.tif", 1/1000),
          ("dist_cbd_km", "grid_dist_cbd_17prov.tif", 1/1000),
          ("dist_dryport_km", "grid_dist_dryport_17prov.tif", 1/1000),
          ("nightlight", "grid_nl_17prov.tif", 1.0),
          ("forest", "grid_forest_2568_17prov.tif", 1.0),
          ("builtup", "grid_builtup_17prov.tif", 1.0),
          ("dist_railway_new_km", "dist_railway_new_km.tif", 1.0)]
FEATURES = [l[0] for l in LAYERS] + ["dist_border_CK_km"]
CK_E, CK_N = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True).transform(100.0894, 20.3622)

# ---- 1. candidates + geometric filters, per province --------------------------------------
cands, manifest = [], []
for _, row in BND.iterrows():
    prov_th, code, poly = row["ADM1_TH"], row["ADM1_PCODE"], row.geometry
    n_test = int((PRES["FPROVNAME"] == prov_th).sum())
    B = max(20000, 200 * n_test)
    rng = np.random.default_rng(SEED)
    minx, miny, maxx, maxy = poly.bounds
    e = rng.uniform(minx, maxx, B)
    n = rng.uniform(miny, maxy, B)
    inside = gpd.GeoSeries(gpd.points_from_xy(e, n), crs="EPSG:32647").within(poly).to_numpy()
    dmin, _ = TREE.query(np.column_stack([e, n]), k=1)
    geo_ok = inside & (dmin >= INNER_M)
    idx = np.flatnonzero(geo_ok)
    cands.append(pd.DataFrame({"ADM1_PCODE": code, "FPROVNAME": prov_th, "draw_index": idx,
                               "easting": e[idx], "northing": n[idx],
                               "dist_nearest_presence_m": dmin[idx]}))
    manifest.append({"ADM1_PCODE": code, "province_th": prov_th, "province_en": row["ADM1_EN"],
                     "n_test": n_test, "budget_B": B, "n_inside_polygon": int(inside.sum()),
                     "n_inside_and_ge_500m": int(geo_ok.sum())})
C = pd.concat(cands, ignore_index=True)

# ---- 2. covariates for every geometric survivor (each raster read once) -------------------
for var, fname, factor in LAYERS:
    with rasterio.open(path(fname)) as src:
        nod = src.nodata if src.nodata is not None else -9999
        r, c = rowcol(src.transform, C["easting"].to_numpy(), C["northing"].to_numpy())
        r, c = np.asarray(r), np.asarray(c)
        ok = (r >= 0) & (r < src.height) & (c >= 0) & (c < src.width)
        a = src.read(1)
        v = np.full(len(C), np.nan)
        v[ok] = a[r[ok], c[ok]].astype(float)
        v[v == nod] = np.nan
        C[var] = v * factor if factor != 1.0 else v
C["dist_border_CK_km"] = np.hypot(C["easting"] - CK_E, C["northing"] - CK_N) / 1000.0
C["complete"] = C[FEATURES].notna().all(axis=1)

# ---- 3. first n_test complete, in draw order; write; record ------------------------------
OUT.mkdir(parents=True, exist_ok=True)
for m in manifest:
    sub = C[(C["ADM1_PCODE"] == m["ADM1_PCODE"]) & C["complete"]].sort_values("draw_index")
    take = sub.head(m["n_test"]).copy()
    m["n_complete_within_budget"] = int(len(sub))
    m["status"] = "complete" if len(take) == m["n_test"] else "incomplete under the locked sampling budget"
    m["last_draw_index_used"] = int(take["draw_index"].max()) if len(take) else None
    take.insert(0, "test_id", [f"tp_{m['ADM1_PCODE']}_{i:04d}" for i in range(len(take))])
    take["presence"] = 0
    cols = ["test_id", "ADM1_PCODE", "FPROVNAME", "draw_index", "easting", "northing",
            "dist_nearest_presence_m"] + FEATURES + ["presence"]
    fn = OUT / f"test_points_{m['ADM1_PCODE']}{'' if m['status'] == 'complete' else '_INCOMPLETE'}.csv"
    take[cols].to_csv(fn, index=False, encoding="utf-8-sig")
    m["file"], m["rows"], m["sha256"] = fn.name, int(len(take)), sha(fn)
    print(f"{m['ADM1_PCODE']} {m['province_en']:<13} n_test={m['n_test']:>4} B={m['budget_B']:>6} "
          f"inside={m['n_inside_polygon']:>6} >=500m={m['n_inside_and_ge_500m']:>6} "
          f"complete={m['n_complete_within_budget']:>6} -> {m['status']} (last j={m['last_draw_index_used']})")

pd.DataFrame(manifest).to_csv(OUT / "test_points_manifest.csv", index=False, encoding="utf-8-sig")
print("total test points:", sum(m["rows"] for m in manifest))
