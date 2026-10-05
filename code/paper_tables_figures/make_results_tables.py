"""Build the Section 4 tables directly from the locked map-stability outputs (read-only).

Writes results_tables.md next to this script. Every number in Section 4 should be traceable to a line here;
nothing is typed by hand. Seed (background-draw) spread is deliberately NOT in these main tables (it goes to
the Supplementary material), per the reporting decision of 2 October 2026.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "exp_map_stability" / "02_data" / "runs"
POST = HERE.parent / "01_posthoc_consensus"
RAD = ["1km", "5km", "10km", "20km", "25km", "Rinf"]
RLAB = {"1km": "1 km", "5km": "5 km", "10km": "10 km", "20km": "20 km", "25km": "25 km", "Rinf": "R∞"}
LRN = ["RF", "XGB", "LR"]
LLAB = {"RF": "RF", "XGB": "XGBoost", "LR": "LR"}
PROV = {"TH50": "Chiang Mai", "TH51": "Lamphun", "TH52": "Lampang", "TH53": "Uttaradit", "TH54": "Phrae",
        "TH55": "Nan", "TH56": "Phayao", "TH57": "Chiang Rai", "TH58": "Mae Hong Son"}
FEAT = {"slope_deg": "Slope", "dist_road_km": "Distance to road", "dist_rail_km": "Distance to existing railway",
        "dist_water_km": "Distance to water", "dist_cbd_km": "Distance to provincial capital",
        "dist_dryport_km": "Distance to dry port", "nightlight": "Nighttime light", "forest": "Forest",
        "builtup": "Built-up", "dist_railway_new_km": "Distance to planned railway",
        "dist_border_CK_km": "Distance to Chiang Khong crossing", "builtup+nightlight": "Built-up + nighttime light (joint)"}
GRP = {"physical_capacity": "Physical capacity", "development_intensity": "Development intensity",
       "transport_access": "Transport access", "forest_area_condition": "Forest condition",
       "market_gateway_access": "Market and gateway access"}
out = []
f3 = lambda x: f"{x:.3f}"


def table(head, rows, align):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join("---:" if a == "r" else "---" for a in align) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    out.append("")


# ---------------------------------------------------------------- Q1: discrimination
fold = pd.read_csv(EXP / "lopo_metrics_by_fold.csv")
assert len(fold) == 9 * 6 * 5 * 3, len(fold)
assert np.abs(fold.tss - fold.kappa).max() < 1e-12
out.append(f"<!-- kappa==TSS: max |TSS-kappa| = {np.abs(fold.tss - fold.kappa).max():.1e} over {len(fold)} rows -->")
# Eq. (3): median over seeds of the across-fold median
per_seed = fold.groupby(["learner", "radius", "seed"])[["auc", "tss"]].median()
summ = per_seed.groupby(["learner", "radius"]).median()
# per-province value = median over seeds; then range across the nine provinces
per_fold = fold.groupby(["learner", "radius", "fold"])[["auc", "tss"]].median()
rng = per_fold.groupby(["learner", "radius"]).agg(["min", "max"])
out.append("### T4 discrimination")
rows = []
for r in RAD:
    row = [RLAB[r]]
    for L in LRN:
        a, t = summ.loc[(L, r)]
        row += [f"{a:.3f} ({rng.loc[(L, r), ('auc', 'min')]:.3f}–{rng.loc[(L, r), ('auc', 'max')]:.3f})",
                f"{t:.3f}"]
    rows.append(row)
table(["Radius"] + [f"{LLAB[L]} {m}" for L in LRN for m in ("AUC", "TSS")], rows, "l" + "rr" * 3)

# ---------------------------------------------------------------- Q2: map agreement
sim = pd.read_csv(EXP / "map_similarity_summary_across_seeds.csv")
assert (sim.n_seeds_compared == 5).all()


def agree(kind, within, a, b):
    s = sim[(sim.comparison == kind) & (sim.within == within) & (sim.a == a) & (sim.b == b)]
    res = []
    for m in ("ssim_raw_7", "ssim_rank_7", "jaccard_top10"):
        v = s[s.metric == m].set_index("province")["median"]          # per province: median over draws
        assert len(v) == 9, (a, b, m, len(v))
        res.append(f"{v.median():.3f} [{v.min():.3f}–{v.max():.3f}]")
    return res


out.append("### T5 across radii (vs 10 km)")
rows = [[LLAB[L], RLAB[r]] + agree("across_radii", L, f"{L}_{r}", f"{L}_10km") for L in LRN for r in RAD if r != "10km"]
table(["Algorithm", "Radius vs 10 km", "SSIM raw", "SSIM rank", "Jaccard top 10%"], rows, "llrrr")
out.append("### T6 across algorithms (same radius)")
pairs = [("RF", "XGB"), ("RF", "LR"), ("XGB", "LR")]
rows = [[RLAB[r], f"{LLAB[a]} vs {LLAB[b]}"] + agree("across_learners", r, f"{a}_{r}", f"{b}_{r}") for r in RAD for a, b in pairs]
table(["Radius", "Pair", "SSIM raw", "SSIM rank", "Jaccard top 10%"], rows, "llrrr")

# 10 km: AUC span vs Jaccard span (headline)
auc10 = [summ.loc[(L, "10km"), "auc"] for L in LRN]
j10 = sim[(sim.comparison == "across_learners") & (sim.within == "10km") & (sim.metric == "jaccard_top10")]
out.append(f"<!-- 10 km: AUC {min(auc10):.4f}-{max(auc10):.4f} span {max(auc10)-min(auc10):.4f}; "
           f"provincial Jaccard (median over draws) {j10['median'].min():.3f}-{j10['median'].max():.3f} -->")
out.append("### T5b per-province Jaccard, 10 km algorithm pairs (median over draws)")
rows = []
for code in PROV:
    rows.append([PROV[code]] + [f3(j10[(j10.a == f"{a}_10km") & (j10.b == f"{b}_10km") & (j10.province == code)]["median"].iloc[0]) for a, b in pairs])
table(["Province", "RF vs XGBoost", "RF vs LR", "XGBoost vs LR"], rows, "lrrr")

# ---------------------------------------------------------------- Q3: factor reliance
sh = pd.read_csv(EXP / "shap" / "shap_summary.csv")
assert sh.groupby(["learner", "radius"]).ngroups == 18
sh["rank"] = sh.groupby(["learner", "radius"])["mean_abs_shap"].rank(ascending=False, method="first")
top3 = sh[sh["rank"] <= 3].groupby("feature").size()
pos = sh.groupby("feature")["direction_spearman"].apply(lambda s: int((s > 0).sum()))
neg = sh.groupby("feature")["direction_spearman"].apply(lambda s: int((s < 0).sum()))
med = sh.groupby(["feature", "learner"])["mean_abs_shap"].median().unstack()
out.append("### T7 SHAP")
order = med.max(axis=1).sort_values(ascending=False).index
rows = [[FEAT[f]] + [f3(med.loc[f, L]) for L in LRN] + [f"{int(top3.get(f, 0))}/18", f"{pos[f]}+ / {neg[f]}−"] for f in order]
table(["Covariate", "RF", "XGBoost", "LR", "Top 3 (configs)", "Direction (+/−)"], rows, "lrrrrr")

perm, lofo = {}, {}
for p in sorted((EXP / "lopo").glob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("status") != "complete":
        continue
    for L, v in d["learners"].items():
        for k, (m, _sd) in v["perm_auc_drop"].items():
            perm.setdefault(k, []).append(m)
        for g, x in v.get("lofo_auc_drop", {}).items():
            lofo.setdefault(g, []).append(x)
out.append("### T8 permutation")
rows = [[FEAT[k], str(len(v)), f3(np.median(v)), f"{np.quantile(v, .25):.3f}–{np.quantile(v, .75):.3f}"]
        for k, v in sorted(perm.items(), key=lambda kv: -np.median(kv[1]))]
table(["Covariate", "n", "Median ΔAUC", "IQR"], rows, "lrrr")
out.append("### T9 LOFO")
rows = [[GRP[k], str(len(v)), f3(np.median(v)), f"{np.quantile(v, .25):.3f}–{np.quantile(v, .75):.3f}"]
        for k, v in sorted(lofo.items(), key=lambda kv: -np.median(kv[1]))]
table(["Group removed", "n", "Median ΔAUC", "IQR"], rows, "lrrr")

# ---------------------------------------------------------------- RQ4: exploratory synthesis
cb = pd.read_csv(POST / "consensus_by_province.csv")
ss = pd.read_csv(POST / "ss_decomposition_by_province.csv")
out.append("### T10 consensus")
rows = []
for code in ["ALL"] + list(PROV):
    a = cb[(cb.set == "A_all90") & (cb.province == code)].iloc[0]
    b = cb[(cb.set == "B_no1km_75") & (cb.province == code)].iloc[0]
    rows.append([("Study area" if code == "ALL" else PROV[code]), f"{a.core_km2:,.1f}", f"{a.dependent_km2:,.1f}",
                 f"{b.core_km2:,.1f}", f"{b.dependent_km2:,.1f}"])
table(["Province", "Core, 90 maps (km²)", "Dependent, 90 maps (km²)", "Core, no 1 km (km²)", "Dependent, no 1 km (km²)"], rows, "lrrrr")
out.append("### T11 decomposition (remainder = draw + interactions)")
rows = []
for code in ["ALL"] + list(PROV):
    r = [("Study area" if code == "ALL" else PROV[code])]
    for st in ("A_all90", "B_no1km_75"):
        x = ss[(ss.set == st) & (ss.province == code)].iloc[0]
        r += [f"{x.share_radius:.2f}", f"{x.share_learner:.2f}", f"{1 - x.share_radius - x.share_learner:.2f}"]
    rows.append(r)
table(["Province", "Radius (90)", "Algorithm (90)", "Remainder (90)", "Radius (no 1 km)", "Algorithm (no 1 km)", "Remainder (no 1 km)"], rows, "lrrrrrr")
one_map = 0.1 * cb[(cb.set == "A_all90") & (cb.province == "ALL")].valid_km2.iloc[0]
out.append(f"<!-- one map top-10% area ~ {one_map:,.1f} km2 -->")

(HERE / "results_tables.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
