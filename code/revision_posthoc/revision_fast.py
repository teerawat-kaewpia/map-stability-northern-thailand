"""Revision post hoc analyses P3, P4, P6, P7 (spec_revision_posthoc_2026-11-01.md, hashed before computing).
No model is fitted here. Reads the locked maps and presence file; writes only to ../outputs/."""
import hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.transform import from_origin, rowcol
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from shapely.ops import unary_union

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
RF = HERE.parents[1]
EXP = RF / "exp_map_stability"
MAPS, RUNS = EXP / "02_data" / "maps", EXP / "02_data" / "runs"
sys.path.insert(0, str(EXP / "01_scripts"))
import run_map_stability as R  # noqa: E402  (locked runner, imported, not edited)

SPEC = HERE / "spec_revision_posthoc_2026-11-01_v1_as_hashed.md"   # spec as hashed before computing (Note 1 appended later to the working copy)
SPEC_SHA = "7434b645295c28cffc7d33e4eff1ba8642353017a3a21da7808d71ace7c9082a"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SPEC) == SPEC_SHA, "spec changed after its hash was recorded"

LOCK = json.loads(R.LOCK.read_text(encoding="utf-8"))
for rec in LOCK["inputs"]:
    assert sha(rec["file"]) == rec["sha256"], f"locked input changed: {rec['file']}"
RAD = [r for r, _ in R.RADII]
valid = np.load(MAPS / "valid_mask.npy"); prov = np.load(MAPS / "province_raster.npy")
H, W = valid.shape; nv = int(valid.sum())
bnd = gpd.read_file([i["file"] for i in LOCK["inputs"] if i["file"].endswith("boundary_9prov_32647.gpkg")][0]) \
    .to_crs("EPSG:32647").sort_values("ADM1_PCODE")
CODES = list(bnd["ADM1_PCODE"])                              # province raster value i+1 -> CODES[i] (runner U1)
G = {"H": H, "W": W, "valid": valid, "prov": prov, "prov_codes": CODES}
load = lambda L, r, s: np.load(MAPS / f"{L}_{r}_s{s}.npy")

# province-order check against the stored hotspot counts (as in posthoc_consensus.py)
sim = pd.read_csv(RUNS / "map_similarity_by_province.csv", encoding="utf-8-sig")
ref = sim[(sim.seed == 42) & (sim.a == "RF_1km") & (sim.b == "RF_10km") & (sim.metric == "jaccard_top10")]
_, cnt = R.jaccard_by_province(G, load("RF", "1km", 42), load("RF", "10km", 42))
for _, r_ in ref.iterrows():
    assert cnt[r_.province]["n_top10_a"] == int(r_.n_top10_a), "province order mismatch"
print("province order verified")


# ---------------------------------------------------------------- P3 structure term of SSIM
def moments(a, b, w=7):
    a = np.nan_to_num(a, nan=0.0).astype("float64"); b = np.nan_to_num(b, nan=0.0).astype("float64")
    cov = w * w / (w * w - 1)                                # skimage use_sample_covariance
    ua, ub = uniform_filter(a, w), uniform_filter(b, w)
    va = cov * (uniform_filter(a * a, w) - ua * ua); vb = cov * (uniform_filter(b * b, w) - ub * ub)
    vab = cov * (uniform_filter(a * b, w) - ua * ub)
    return ua, ub, np.clip(va, 0, None), np.clip(vb, 0, None), vab


C1, C2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
C3 = C2 / 2
winvalid = uniform_filter(valid.astype("float64"), size=7, mode="constant", cval=0.0) >= 1 - 1e-9


def by_prov(S):
    return {c: float(S[(prov == i + 1) & winvalid].mean()) for i, c in enumerate(CODES)}


def ssim_terms(va_, vb_):
    ua, ub, va, vb, vab = moments(R.to_grid(G, va_), R.to_grid(G, vb_))
    full = ((2 * ua * ub + C1) * (2 * vab + C2)) / ((ua ** 2 + ub ** 2 + C1) * (va + vb + C2))
    struct = (vab + C3) / (np.sqrt(va) * np.sqrt(vb) + C3)
    return by_prov(full), by_prov(struct)


rk = lambda v: rankdata(v, method="average") / len(v)
# implementation check: full rank SSIM from these moments must reproduce the stored values
chk = sim[(sim.seed == 42) & (sim.a == "RF_10km") & (sim.b == "LR_10km") & (sim.metric == "ssim_rank_7")]
full_chk, _ = ssim_terms(rk(load("RF", "10km", 42)), rk(load("LR", "10km", 42)))
dev = max(abs(full_chk[p] - v) for p, v in zip(chk.province, chk.value))
print(f"P3 implementation check (rank SSIM RF_10km vs LR_10km, s42): max |diff| = {dev:.2e}")
assert dev < 1e-6, "moment implementation does not reproduce the stored rank SSIM"

pairs = [("across_radii", L, (L, r), (L, R.REF_RADIUS)) for L in R.LEARNERS for r in RAD if r != R.REF_RADIUS] + \
        [("across_learners", r, (a, r), (b, r)) for r in RAD for a, b in [("RF", "XGB"), ("RF", "LR"), ("XGB", "LR")]]
rows = []
for seed in R.SEEDS:
    for kind, within, (La, ra), (Lb, rb) in pairs:
        f_, s_ = ssim_terms(rk(load(La, ra, seed)), rk(load(Lb, rb, seed)))
        for c in CODES:
            rows.append({"seed": seed, "comparison": kind, "within": within, "a": f"{La}_{ra}", "b": f"{Lb}_{rb}",
                         "province": c, "ssim_rank_7_full": f_[c], "ssim_rank_7_structure": s_[c]})
p3 = pd.DataFrame(rows); p3.to_csv(OUT / "P3_ssim_structure_by_province.csv", index=False)
p3s = (p3.groupby(["comparison", "within", "a", "b", "province"])[["ssim_rank_7_full", "ssim_rank_7_structure"]]
         .median().groupby(["comparison", "within", "a", "b"]).agg(["median", "min", "max"]))
p3s.columns = ["_".join(c) for c in p3s.columns]; p3s.reset_index().to_csv(OUT / "P3_ssim_structure_summary.csv", index=False)
print("P3 done:", len(p3), "rows")

# ---------------------------------------------------------------- P4 between-draw agreement at 10 km
rows = []
from itertools import combinations
for L in R.LEARNERS:
    for s1, s2 in combinations(R.SEEDS, 2):
        a, b = load(L, "10km", s1), load(L, "10km", s2)
        raw = R.ssim_by_province(G, R.to_grid(G, a), R.to_grid(G, b), 7)
        rnk = R.ssim_by_province(G, R.to_grid(G, rk(a)), R.to_grid(G, rk(b)), 7)
        jac, _ = R.jaccard_by_province(G, a, b)
        for c in CODES:
            rows.append({"learner": L, "seed_a": s1, "seed_b": s2, "province": c,
                         "ssim_raw_7": raw[c], "ssim_rank_7": rnk[c], "jaccard_top10": jac[c]})
p4 = pd.DataFrame(rows); p4.to_csv(OUT / "P4_between_draw_10km_by_province.csv", index=False)
p4s = (p4.groupby(["learner", "province"])[["ssim_raw_7", "ssim_rank_7", "jaccard_top10"]].median()
         .groupby("learner").agg(["median", "min", "max"]))
p4s.columns = ["_".join(c) for c in p4s.columns]; p4s.reset_index().to_csv(OUT / "P4_between_draw_10km_summary.csv", index=False)
print("P4 done:\n", p4s.round(3).to_string())

# ---------------------------------------------------------------- P6 variance components of hotspot membership
I = np.zeros((len(RAD), len(R.LEARNERS), len(R.SEEDS), nv), dtype=np.uint8)
for i, r in enumerate(RAD):
    for j, L in enumerate(R.LEARNERS):
        for k, s in enumerate(R.SEEDS):
            v = load(L, r, s); I[i, j, k] = v >= np.quantile(v, 0.9)
pv = prov[valid]
rows = []
for set_name, ridx in [("A_all90", list(range(6))), ("B_no1km_75", list(range(1, 6)))]:
    X = I[ridx].astype(np.float64)                            # (R, L, D, N)
    Rn, Ln, Dn = X.shape[:3]
    m = X.mean(axis=(0, 1, 2))
    f = m
    mr, ml, md = X.mean(axis=(1, 2)), X.mean(axis=(0, 2)), X.mean(axis=(0, 1))
    mrl, mrd, mld = X.mean(axis=2), X.mean(axis=1), X.mean(axis=0)
    ss = {"T": ((X - m) ** 2).sum(axis=(0, 1, 2)),
          "R": Ln * Dn * ((mr - m) ** 2).sum(axis=0),
          "L": Rn * Dn * ((ml - m) ** 2).sum(axis=0),
          "D": Rn * Ln * ((md - m) ** 2).sum(axis=0),
          "RL": Dn * ((mrl - mr[:, None] - ml[None] + m) ** 2).sum(axis=(0, 1)),
          "RD": Ln * ((mrd - mr[:, None] - md[None] + m) ** 2).sum(axis=(0, 1)),
          "LD": Rn * ((mld - ml[:, None] - md[None] + m) ** 2).sum(axis=(0, 1))}
    ss["RLD"] = ss["T"] - sum(ss[k] for k in ("R", "L", "D", "RL", "RD", "LD"))
    df1 = {"R": Rn - 1, "L": Ln - 1, "D": Dn - 1, "RL": (Rn - 1) * (Ln - 1), "RD": (Rn - 1) * (Dn - 1),
           "LD": (Ln - 1) * (Dn - 1), "RLD": (Rn - 1) * (Ln - 1) * (Dn - 1)}
    dep = (f > 0) & (f < 0.9)
    groups = [("ALL", np.ones(nv, bool))] + [(c, pv == i + 1) for i, c in enumerate(CODES)]
    for code, g in groups:
        sel = dep & g; n = int(sel.sum())
        if n == 0:
            continue
        S = {k: float(v[sel].sum()) for k, v in ss.items()}
        MS = {k: S[k] / (df1[k] * n) for k in df1}
        comp = {"R": max((MS["R"] - MS["RD"]) / (Ln * Dn), 0), "L": max((MS["L"] - MS["LD"]) / (Rn * Dn), 0),
                "RL": max((MS["RL"] - MS["RLD"]) / Dn, 0), "D": max((MS["D"] - MS["RLD"]) / (Rn * Ln), 0),
                "RD": max((MS["RD"] - MS["RLD"]) / Ln, 0), "LD": max((MS["LD"] - MS["RLD"]) / Rn, 0),
                "res_RLD": MS["RLD"]}
        tot = sum(comp.values())
        row = {"set": set_name, "province": code, "dependent_cells": n}
        row.update({f"ss_share_{k}": S[k] / S["T"] for k in df1})
        row.update({f"df_{k}": df1[k] for k in df1})
        row.update({f"MS_{k}": MS[k] for k in df1})
        row.update({f"vc_{k}": comp[k] for k in comp})
        row.update({f"vc_share_{k}": comp[k] / tot for k in comp})
        rows.append(row)
p6 = pd.DataFrame(rows); p6.to_csv(OUT / "P6_variance_components.csv", index=False)
show = p6[p6.province == "ALL"][["set"] + [c for c in p6.columns if c.startswith(("ss_share", "vc_share"))]]
print("P6 done:\n", show.round(3).T.to_string())

# ---------------------------------------------------------------- P7 confirmed vs unchecked presences
pres = pd.read_csv([i["file"] for i in LOCK["inputs"] if i["file"].endswith("analysis_dataset_m3_base_clean.csv")][0],
                   encoding="utf-8-sig")
code_of = dict(zip(bnd["ADM1_TH"], bnd["ADM1_PCODE"])); en_of = dict(zip(bnd["ADM1_PCODE"], bnd["ADM1_EN"]))
pres["code"] = pres["FPROVNAME"].map(code_of)
union = unary_union(list(bnd.geometry)); minx, _, _, maxy = union.bounds
tr = from_origin(np.floor(minx / 500) * 500, np.ceil(maxy / 500) * 500, 500.0, 500.0)   # runner U1 grid
r_, c_ = rowcol(tr, pres.easting.to_numpy(), pres.northing.to_numpy()); r_, c_ = np.asarray(r_), np.asarray(c_)
inside = (r_ >= 0) & (r_ < H) & (c_ >= 0) & (c_ < W)
idx_grid = np.full(valid.shape, -1, int); idx_grid[valid] = np.arange(nv)
cell = np.full(len(pres), -1); cell[inside] = idx_grid[r_[inside], c_[inside]]
for L in R.LEARNERS:
    v = load(L, "10km", 42); top = v >= np.quantile(v, 0.9)
    pres[f"score_{L}_10km_s42"] = np.where(cell >= 0, v[np.clip(cell, 0, None)], np.nan)
    pres[f"top10_{L}_10km_s42"] = np.where(cell >= 0, top[np.clip(cell, 0, None)], np.nan)
conf = pres[pres.source == "confirmed"]; unc = pres[pres.source == "factory"]
d_nn, _ = cKDTree(conf[["easting", "northing"]].to_numpy()).query(unc[["easting", "northing"]].to_numpy())
pres.loc[unc.index, "dist_to_nearest_confirmed_km"] = d_nn / 1000
rows = []
for code, g in [("ALL", pres)] + [(c, pres[pres.code == c]) for c in CODES]:
    for src, gg in g.groupby("source"):
        row = {"province": code, "name": en_of.get(code, "Corridor"), "group": src, "n": len(gg),
               "cells_off_grid": int((cell[gg.index] < 0).sum())}
        row.update({f"median_{f}": float(gg[f].median()) for f in R.FEATURES})
        for L in R.LEARNERS:
            row[f"median_score_{L}_10km"] = float(gg[f"score_{L}_10km_s42"].median())
            row[f"share_top10_{L}_10km"] = float(gg[f"top10_{L}_10km_s42"].mean())
        if src == "factory":
            row["median_dist_to_nearest_confirmed_km"] = float(gg["dist_to_nearest_confirmed_km"].median())
            row["share_within_1km_of_confirmed"] = float((gg["dist_to_nearest_confirmed_km"] <= 1).mean())
        rows.append(row)
p7 = pd.DataFrame(rows); p7.to_csv(OUT / "P7_confirmed_vs_unchecked.csv", index=False)
print("P7 done:\n", p7[p7.province == "ALL"].T.to_string())

json.dump({"spec_sha256": SPEC_SHA, "outputs": {p.name: sha(p) for p in sorted(OUT.glob("P[3467]_*.csv"))}},
          open(OUT / "fast_outputs_sha256.json", "w"), indent=1)
print("all fast analyses done")
