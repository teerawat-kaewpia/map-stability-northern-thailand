"""POST-HOC sensitivity analyses (spec: posthoc_sensitivity_spec_2026-10-05.md; its SHA-256 is checked first).
No model is refitted; inputs are files saved by the locked run (2026-10-01) and the consensus analysis (2026-10-02).
Firm coordinates are used in S4 only to compute distances and are never written out."""
import hashlib
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
PD = HERE.parent
RF = PD.parent
EXP = RF / "exp_map_stability"
spec = HERE / "posthoc_sensitivity_spec_2026-10-05.md"
want = (HERE / "posthoc_sensitivity_spec_2026-10-05.sha256").read_text().split()[0]
assert hashlib.sha256(spec.read_bytes()).hexdigest() == want, "spec changed after it was hashed"

ALG = ["RF", "XGB", "LR"]
RAD = ["1km", "5km", "10km", "20km", "25km", "Rinf"]
DRAWS = [42, 0, 1, 7, 2024]
PAIRS = [("RF", "XGB"), ("RF", "LR"), ("XGB", "LR")]
out = {}

# ---------------------------------------------------------------- S1 paired AUC differences, aggregation order
f = pd.read_csv(EXP / "02_data/runs/lopo_metrics_by_fold.csv", encoding="utf-8-sig")
w = f.pivot_table(index=["fold", "radius", "seed"], columns="learner", values="auc").reset_index()
rows = []
for r in RAD:
    s = w[w.radius == r]
    for a, b in PAIRS:
        d = (s[a] - s[b]).groupby(s["fold"]).median()
        rows.append({"comparison": "algorithm pair", "radius": r, "a": a, "b": b, "median_dAUC": d.median(),
                     "min_dAUC": d.min(), "max_dAUC": d.max(), "folds_a_higher": int((d > 0).sum()), "folds": len(d)})
for L in ALG:
    p = f[f.learner == L].pivot_table(index=["fold", "seed"], columns="radius", values="auc")
    for r in RAD:
        if r == "10km":
            continue
        d = (p[r] - p["10km"]).groupby(level="fold").median()
        rows.append({"comparison": "radius vs 10 km", "radius": r, "a": f"{L}_{r}", "b": f"{L}_10km",
                     "median_dAUC": d.median(), "min_dAUC": d.min(), "max_dAUC": d.max(),
                     "folds_a_higher": int((d > 0).sum()), "folds": len(d)})
s1a = pd.DataFrame(rows)
s1a.to_csv(HERE / "S1_paired_auc_differences.csv", index=False)
agg = []
for (L, r), g in f.groupby(["learner", "radius"]):
    a_ = g.groupby("seed").auc.median().median()
    b_ = g.groupby("fold").auc.median().median()
    agg.append({"learner": L, "radius": r, "a_med_draws_of_med_folds": a_, "b_med_folds_of_med_draws": b_,
                "c_pooled_median": g.auc.median(), "d_pooled_mean": g.auc.mean()})
s1b = pd.DataFrame(agg)
s1b["max_abs_diff_from_a"] = s1b[["b_med_folds_of_med_draws", "c_pooled_median", "d_pooled_mean"]].sub(
    s1b.a_med_draws_of_med_folds, axis=0).abs().max(axis=1)
s1b.to_csv(HERE / "S1_aggregation_order.csv", index=False)
claims = {}
for col in ["a_med_draws_of_med_folds", "b_med_folds_of_med_draws", "c_pooled_median", "d_pooled_mean"]:
    ge5 = s1b[s1b.radius != "1km"][col]
    ten = s1b[s1b.radius == "10km"][col]
    claims[col] = {"range_5km_plus": [round(ge5.min(), 4), round(ge5.max(), 4)],
                   "within_0.970_0.976_at_3dp": bool(round(ge5.min(), 3) >= 0.970 and round(ge5.max(), 3) <= 0.976),
                   "spread_10km": round(ten.max() - ten.min(), 5), "spread_10km_le_0.004": bool(ten.max() - ten.min() <= 0.004)}
out["S1"] = {"aggregation_claims": claims, "max_abs_diff_any_order": round(s1b.max_abs_diff_from_a.max(), 5)}

# ---------------------------------------------------------------- S2 hotspot counts, ties, cut-offs
ms = pd.read_csv(EXP / "02_data/runs/map_similarity_by_province.csv", encoding="utf-8-sig")
c10 = ms[(ms.metric == "jaccard_top10") & (ms.comparison == "across_learners") & (ms.within == "10km")]
cnt = c10.groupby(["a", "b", "province"])[["n_top10_a", "n_top10_b", "n_intersection", "n_union", "value"]].median().reset_index()
cnt.rename(columns={"value": "jaccard"}).to_csv(HERE / "S2_hotspot_counts_10km_algorithm_pairs.csv", index=False)
valid = np.load(EXP / "02_data/maps/valid_mask.npy")
prov = np.load(EXP / "02_data/maps/province_raster.npy")[valid]
codes = [f"TH{50 + i}" for i in range(9)]
maps = {(L, r, d): np.load(EXP / f"02_data/maps/{L}_{r}_s{d}.npy") for L in ALG for r in RAD for d in DRAWS}
n = int(valid.sum())
ties = []
for k, v in maps.items():
    q = np.quantile(v, 0.9)
    ties.append({"map": f"{k[0]}_{k[1]}_s{k[2]}", "cells_equal_threshold": int((v == q).sum()),
                 "hotspot_cells": int((v >= q).sum()), "excess_over_10pct": int((v >= q).sum() - round(0.1 * n))})
ties = pd.DataFrame(ties)
ties.to_csv(HERE / "S2_ties_at_top10.csv", index=False)
out["S2_ties"] = {"max_cells_equal_threshold": int(ties.cells_equal_threshold.max()),
                  "max_excess_over_10pct": int(ties.excess_over_10pct.max()), "valid_cells": n}


def jac(va, vb, p):
    ta, tb = va >= np.quantile(va, 1 - p), vb >= np.quantile(vb, 1 - p)
    res = {}
    for i, c in enumerate(codes):
        m = prov == i + 1
        u = ((ta | tb) & m).sum()
        res[c] = ((ta & tb) & m).sum() / u if u else np.nan
    return res


comps = [("radius vs 10 km", L, (L, r), (L, "10km")) for L in ALG for r in RAD if r != "10km"] + \
        [("algorithm pair", r, (a, r), (b, r)) for r in RAD for a, b in PAIRS]
jrows = []
for p in (0.05, 0.10, 0.20):
    for kind, within, A, B in comps:
        per = pd.DataFrame([jac(maps[(A[0], A[1], d)], maps[(B[0], B[1], d)], p) for d in DRAWS])
        provmed = per.median(axis=0)
        jrows.append({"top_share": p, "comparison": kind, "within": within, "a": f"{A[0]}_{A[1]}", "b": f"{B[0]}_{B[1]}",
                      "median_over_provinces": provmed.median(), "min_province": provmed.min(), "max_province": provmed.max()})
jr = pd.DataFrame(jrows)
jr.to_csv(HERE / "S2_jaccard_by_cutoff.csv", index=False)
# audit: 10% must reproduce the run
run = ms[ms.metric == "jaccard_top10"].groupby(["a", "b", "province"]).value.median().reset_index()
runmed = run.groupby(["a", "b"]).value.median()
j10 = jr[jr.top_share == 0.10].set_index(["a", "b"]).median_over_provinces
common = runmed.index.intersection(j10.index)
out["S2_audit_top10_max_abs_diff"] = float((runmed.loc[common] - j10.loc[common]).abs().max())
fd = pd.read_csv(PD / "01_posthoc_consensus/f_distribution.csv")
fa = []
for s_, nm in (("A_all90", 90), ("B_no1km_75", 75)):
    g = fd[fd.set == s_]
    fa.append({"set": s_, "km2_f_ge_0.8": g.loc[g.f >= 0.8 - 1e-9, "cells"].sum() * 0.25,
               "km2_f_ge_0.9": g.loc[g.f >= 0.9 - 1e-9, "cells"].sum() * 0.25,
               "km2_f_eq_1": g.loc[g.n_maps_hotspot == nm, "cells"].sum() * 0.25})
pd.DataFrame(fa).to_csv(HERE / "S2_consensus_by_f_cutoff.csv", index=False)
out["S2_consensus"] = fa

# ---------------------------------------------------------------- S3 radius x algorithm interaction
s3 = []
for setname, rads in (("A_all90", RAD), ("B_no1km_75", [r for r in RAD if r != "1km"])):
    I = np.stack([np.stack([np.stack([maps[(L, r, d)] >= np.quantile(maps[(L, r, d)], 0.9) for d in DRAWS])
                            for L in ALG]) for r in rads]).astype(np.float32)          # R x L x D x cells
    R_, L_, D_ = I.shape[:3]
    fq = I.mean(axis=(0, 1, 2))
    dep = (fq > 0) & (fq < 0.9 - 1e-9)
    I = I[..., dep]
    m = I.mean(axis=(0, 1, 2))
    mr, ml, md = I.mean(axis=(1, 2)), I.mean(axis=(0, 2)), I.mean(axis=(0, 1))
    mrl = I.mean(axis=2)
    ss_t = ((I - m) ** 2).sum()
    ss_r = L_ * D_ * ((mr - m) ** 2).sum()
    ss_l = R_ * D_ * ((ml - m) ** 2).sum()
    ss_d = R_ * L_ * ((md - m) ** 2).sum()
    ss_rl = D_ * ((mrl - mr[:, None, :] - ml[None, :, :] + m) ** 2).sum()
    s3.append({"set": setname, "dependent_cells": int(dep.sum()), "share_radius": ss_r / ss_t, "share_algorithm": ss_l / ss_t,
               "share_draw": ss_d / ss_t, "share_radius_x_algorithm": ss_rl / ss_t,
               "share_interactions_with_draw": (ss_t - ss_r - ss_l - ss_d - ss_rl) / ss_t})
pd.DataFrame(s3).to_csv(HERE / "S3_radius_x_algorithm.csv", index=False)
out["S3"] = s3

# ---------------------------------------------------------------- S4 spatial diagnostics
tp = pd.concat([pd.read_csv(x, encoding="utf-8-sig") for x in sorted((EXP / "02_data/test_points").glob("test_points_TH*.csv"))])
name2code = dict(zip(tp.FPROVNAME, tp.ADM1_PCODE))
pres = pd.read_csv(RF / "data/processed/analysis_dataset_m3_base_clean.csv", encoding="utf-8-sig")
pres["code"] = pres.FPROVNAME.map(name2code)
assert pres.code.notna().all() and len(pres) == 2859
s4 = []
for c in codes:
    test_p = pres[pres.code == c][["easting", "northing"]].to_numpy()
    train_p = pres[pres.code != c][["easting", "northing"]].to_numpy()
    test_b = tp[tp.ADM1_PCODE == c][["easting", "northing"]].to_numpy()
    tree = cKDTree(train_p)
    dp, db = tree.query(test_p)[0] / 1000, tree.query(test_b)[0] / 1000
    s4.append({"fold": c, "n_test_presences": len(test_p), "n_test_background": len(test_b),
               "presence_to_train_km_p10": np.percentile(dp, 10), "presence_to_train_km_median": np.median(dp),
               "presence_to_train_km_p90": np.percentile(dp, 90), "background_to_train_km_p10": np.percentile(db, 10),
               "background_to_train_km_median": np.median(db), "background_to_train_km_p90": np.percentile(db, 90)})
pd.DataFrame(s4).to_csv(HERE / "S4_train_test_distances.csv", index=False)
pre = (EXP / "00_predeclaration/predeclaration_2026-10-01.md").read_text(encoding="utf-8")
mm = re.findall(r"^\| \d{14} \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", pre, flags=re.M)
assert len(mm) == 6, mm
pd.DataFrame([{"record": i + 1, "registry_province": a.strip(), "lies_in_polygon_of": b.strip(),
               "distance_to_polygon_edge": d.strip()} for i, (a, b, d) in enumerate(mm)]).to_csv(
    HERE / "S4_registry_polygon_mismatches.csv", index=False)
# fold map (no firm locations)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
bnd = gpd.read_file(RF / "paper_B/03_results/rasters/boundary_9prov_32647.gpkg").to_crs(32647).sort_values("ADM1_PCODE")
fig, ax = plt.subplots(figsize=(5.6, 5.4))
cols = ["#cde2fb", "#f6d7b0", "#d5ecd4", "#e6d6f2", "#fbe3e8", "#d9eef2", "#f2ead0", "#e1e4f5", "#eadfd3"]
bnd.plot(ax=ax, color=cols, edgecolor="#4a4a4a", linewidth=0.6)
s4d = {x["fold"]: x for x in s4}
for _, r in bnd.iterrows():
    pnt = r.geometry.representative_point()
    ax.text(pnt.x, pnt.y, f"{r.ADM1_PCODE}\n{r.ADM1_EN}\nn = {s4d[r.ADM1_PCODE]['n_test_presences']}", ha="center",
            va="center", fontsize=6.5, color="#222222")
x0, y0, x1, y1 = bnd.total_bounds
ax.plot([x0 + 5000, x0 + 55000], [y0 + 12000, y0 + 12000], color="#222222", lw=2)
ax.text(x0 + 30000, y0 + 18000, "50 km", ha="center", fontsize=7)
ax.annotate("N", xy=(x1 - 15000, y1 - 5000), xytext=(x1 - 15000, y1 - 35000), ha="center", fontsize=8,
            arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1))
ax.set_aspect("equal"); ax.set_axis_off()
fig.savefig(HERE / "figS1_fold_map.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
out["S4"] = {"rows": len(s4)}
(HERE / "posthoc_sensitivity_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(json.dumps(out, indent=1, default=float)[:3000])
