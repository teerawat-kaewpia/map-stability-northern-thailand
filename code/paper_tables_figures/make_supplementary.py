"""Build supplementary_src.md: S1-S8 = tables moved out of the main text (a figure shows the same data there);
S9-S13 = tables generated here, read-only from the locked experiment outputs. Nothing typed by hand."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "exp_map_stability"
RUNS = EXP / "02_data" / "runs"
RAD = ["1km", "5km", "10km", "20km", "25km", "Rinf"]
RL = {"1km": "1 km", "5km": "5 km", "10km": "10 km", "20km": "20 km", "25km": "25 km", "Rinf": "R∞"}
LRN = ["RF", "XGB", "LR"]
LN = {"RF": "RF", "XGB": "XGBoost", "LR": "LR"}
FEAT = {"slope_deg": "Slope", "dist_road_km": "Distance to road", "dist_rail_km": "Distance to existing railway",
        "dist_water_km": "Distance to water", "dist_cbd_km": "Distance to provincial capital",
        "dist_dryport_km": "Distance to dry port", "nightlight": "Nighttime light", "forest": "Forest",
        "builtup": "Built-up", "dist_railway_new_km": "Distance to planned railway",
        "dist_border_CK_km": "Distance to Chiang Khong crossing", "builtup+nightlight": "Built-up + nighttime light (joint)"}
GRP = {"physical_capacity": "Physical capacity", "development_intensity": "Development intensity",
       "transport_access": "Transport access", "forest_area_condition": "Forest condition",
       "market_gateway_access": "Market and gateway access"}
PROV = {"TH50": "Chiang Mai", "TH51": "Lamphun", "TH52": "Lampang", "TH53": "Uttaradit", "TH54": "Phrae",
        "TH55": "Nan", "TH56": "Phayao", "TH57": "Chiang Rai", "TH58": "Mae Hong Son"}
out = []


def table(cap, head, rows, align):
    out.append(cap + "\n")
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join("---:" if a == "r" else "---" for a in align) + "|")
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    out.append("")


out.append("# Supplementary material\n")
out.append("**Comparable Accuracy, Divergent Hotspots: A Pre-specified Crossed Experiment on Background Radius and "
           "Algorithm Choice in Industrial-Location Suitability Mapping, Northern Thailand**\n")
out.append("Tables S1–S8 give in tabular form the data that the main text presents as figures (Table S1: the presence data mapped in Figure 1). Tables S9 and S10 give the "
           "values behind the factor-reliance-by-ring figures. Table S11 gives the spread of held-out "
           "discrimination across the five background draws, Table S12 the map-agreement results for the "
           "21 × 21 SSIM window, and Table S13 a post-hoc comparison of between-algorithm and between-draw "
           "hotspot overlap.\n")
out.append((HERE / "_moved_tables.md").read_text(encoding="utf-8").strip() + "\n")

# S9 SHAP by ring
sh = pd.read_csv(RUNS / "shap" / "shap_summary.csv")
assert sh.groupby(["learner", "radius"]).ngroups == 18
rows = []
order = sh.groupby("feature")["mean_abs_shap"].median().sort_values(ascending=False).index
for L in LRN:
    for f in order:
        s = sh[(sh.learner == L) & (sh.feature == f)].set_index("radius")
        rows.append([LN[L], FEAT[f]] + [f"{s.loc[r, 'mean_abs_shap'] * 100:.1f}{'+' if s.loc[r, 'direction_spearman'] > 0 else '−'}"
                                        for r in RAD])
table("**Table S9.** Mean absolute SHAP value × 100 (probability scale) by algorithm, covariate and background ring, "
      "with the sign of the covariate–SHAP Spearman correlation (+ or −). Map models of the first pre-declared "
      "background draw, evaluated on the same 1,000 cells; logistic-regression values are sampled approximations.",
      ["Algorithm", "Covariate"] + [RL[r] for r in RAD], rows, "ll" + "r" * 6)

# S10 permutation and LOFO by ring
rec = []
for p in sorted((RUNS / "lopo").glob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("status") != "complete":
        continue
    for L, v in d["learners"].items():
        for k, (m, _sd) in v["perm_auc_drop"].items():
            rec.append(("perm", d["radius"], k, m))
        for g, x in v.get("lofo_auc_drop", {}).items():
            rec.append(("lofo", d["radius"], g, x))
pr = pd.DataFrame(rec, columns=["kind", "radius", "name", "drop"])
assert (pr[pr.kind == "perm"].groupby(["name", "radius"]).size() == 135).all()
assert (pr[pr.kind == "lofo"].groupby(["name", "radius"]).size() == 27).all()
med = pr.groupby(["kind", "name", "radius"])["drop"].median()
rows = [["*Permutation (n = 135 per cell)*"] + [""] * 6]
for k in med.loc["perm"].groupby("name").median().sort_values(ascending=False).index:
    rows.append([FEAT[k]] + [f"{med.loc[('perm', k, r)]:.3f}" for r in RAD])
rows.append(["*Group removed and refitted (n = 27 per cell)*"] + [""] * 6)
for g in med.loc["lofo"].groupby("name").median().sort_values(ascending=False).index:
    rows.append([GRP[g]] + [f"{med.loc[('lofo', g, r)]:.3f}" for r in RAD])
table("**Table S10.** Median loss of held-out AUC by background ring. Permutation: all folds, draws and "
      "algorithms. Group removal: first pre-declared draw, all folds and algorithms.",
      ["Covariate or group"] + [RL[r] for r in RAD], rows, "l" + "r" * 6)

# S11 spread across draws
sp = pd.read_csv(RUNS / "lopo_fold_spread_by_seed.csv")
rows = []
for r in RAD:
    row = [RL[r]]
    for L in LRN:
        s = sp[(sp.learner == L) & (sp.radius == r)]
        assert len(s) == 5
        row += [f"{s.auc_median.median():.3f} ({s.auc_median.min():.3f}–{s.auc_median.max():.3f})",
                f"{s.tss_median.median():.3f} ({s.tss_median.min():.3f}–{s.tss_median.max():.3f})"]
    rows.append(row)
table("**Table S11.** Held-out discrimination across the five background draws: median over draws of the "
      "median across the nine province folds, with the range of the five draw-level values in parentheses. "
      "These are ranges across draws, not confidence intervals.",
      ["Radius"] + [f"{LN[L]} {m}" for L in LRN for m in ("AUC", "TSS")], rows, "l" + "rr" * 3)

# S12 SSIM 21 x 21
sim = pd.read_csv(RUNS / "map_similarity_summary_across_seeds.csv")


def ag(kind, within, a, b, m):
    v = sim[(sim.comparison == kind) & (sim.within == within) & (sim.a == a) & (sim.b == b) & (sim.metric == m)]["median"]
    assert len(v) == 9
    return f"{v.median():.3f} [{v.min():.3f}–{v.max():.3f}]"


rows = [[LN[L], f"{RL[r]} vs 10 km", ag("across_radii", L, f"{L}_{r}", f"{L}_10km", "ssim_raw_21"),
         ag("across_radii", L, f"{L}_{r}", f"{L}_10km", "ssim_rank_21")] for L in LRN for r in RAD if r != "10km"]
rows += [[f"{LN[a]} vs {LN[b]}", RL[r], ag("across_learners", r, f"{a}_{r}", f"{b}_{r}", "ssim_raw_21"),
          ag("across_learners", r, f"{a}_{r}", f"{b}_{r}", "ssim_rank_21")]
         for r in RAD for a, b in (("RF", "XGB"), ("RF", "LR"), ("XGB", "LR"))]
table("**Table S12.** Sensitivity of SSIM to window size: 21 × 21 cells (10.5 km) instead of 7 × 7. Median of the "
      "nine provincial values, each a median over the five background draws, with the provincial range in brackets. "
      "Upper rows: each radius against the 10 km map within one algorithm; lower rows: pairs of algorithms at the "
      "same radius.", ["Comparison", "Radius", "SSIM raw", "SSIM rank"], rows, "llrr")

# S13 post-hoc between-algorithm vs between-draw overlap (10 km)
ph = pd.read_csv(EXP / "04_notes" / "posthoc_jaccard_seed_vs_learner_2026-10-02.csv")
cols = list(ph.columns)
kind_col, prov_col, pair_col, val_col = cols[0], cols[1], cols[2], cols[-1]
rows, lower = [], 0
for code in PROV:
    for pair in ("RF-XGB", "RF-LR", "XGB-LR"):
        a, b = pair.split("-")
        btw = ph[(ph[kind_col] == "between_learners_same_seed") & (ph[prov_col] == code) & (ph[pair_col] == pair)][val_col]
        wit = ph[(ph[kind_col] != "between_learners_same_seed") & (ph[prov_col] == code)
                 & (ph[cols[3]].isin([a, b]))][val_col]
        assert len(btw) == 5 and len(wit) == 20, (code, pair, len(btw), len(wit))
        lower += btw.median() < wit.median()
        rows.append([PROV[code], f"{LN[a]} vs {LN[b]}", f"{btw.median():.3f}", f"{wit.median():.3f}"])
table(f"**Table S13.** Post-hoc comparison at the 10 km radius: hotspot overlap (Jaccard, top 10%) between "
      f"algorithms at the same background draw (5 values per cell) against overlap between draws within the same "
      f"algorithm (10 draw pairs for each of the two algorithms; 20 values per cell). The between-algorithm median "
      f"is lower in {lower} of 27 cells. This comparison was defined after the confirmatory results had been read; it "
      f"is descriptive and involves no significance test.",
      ["Province", "Algorithm pair", "Between algorithms", "Between draws"], rows, "llrr")

# ---- MDPI pass (2026-10-04): tables renumbered in order of first citation in the manuscript, comma only in numbers
# of five or more digits, American spelling; the manuscript's Supplementary Materials list is written from here.
import re
text = "\n".join(out)
head, *blocks = re.split(r"\n(?=\*\*Table S\d+\.\*\*)", text)
mp = dict(l.split(" -> ") for l in (HERE / "_supp_renumber_map.txt").read_text(encoding="utf-8").splitlines())
renum = {}
for b in blocks:
    old = re.match(r"\*\*Table (S\d+)\.\*\*", b).group(1)
    renum[int(mp[old][1:])] = re.sub(r"^\*\*Table S\d+\.\*\*", f"**Table {mp[old]}.**", b)
head = re.sub(r"Tables S1–S8 give.*?hotspot overlap\.\n",
              "Tables S1–S13 give the tables cited in the main text, numbered in the order in which they are first "
              "cited.\n", head, flags=re.S)
text = head + "\n" + "\n".join(renum[k] for k in sorted(renum))
text = re.sub(r"(?<![\d.,])(\d),(\d{3})(?![\d,])", r"\1\2", text)
for k, v in {"summarised": "summarized", "modelling": "modeling", "standardised": "standardized",
             "licence": "license", "normalised": "normalized"}.items():
    text = re.sub(rf"\b{k}\b", v, text)
(HERE / "supplementary_src.md").write_text(text, encoding="utf-8")
titles = []
for k in sorted(renum):
    cap = re.match(r"\*\*Table S\d+\.\*\* (.+?)(?<!e\.g)(?<!i\.e)\.(?:\s|$)", renum[k]).group(1)
    titles.append(f"Table S{k}: {cap}")
(HERE / "_supp_titles.txt").write_text("\n".join(titles), encoding="utf-8")
print("written; between<within in", lower, "of 27;", len(titles), "tables")
