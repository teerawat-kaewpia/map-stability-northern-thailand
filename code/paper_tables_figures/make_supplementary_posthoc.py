"""Append the POST-HOC sensitivity tables S14-S20 and Figure S1 to supplementary_src.md (idempotent: an earlier
appended block is replaced). Source: Paper_D/06_posthoc_sensitivity/ (spec hashed 2026-10-05 before computing)."""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PH = HERE.parent / "06_posthoc_sensitivity"
SUP = HERE / "supplementary_src.md"
MARK = "<!-- POSTHOC-SENSITIVITY-START -->"
ALG = {"RF": "Random forest", "XGB": "XGBoost", "LR": "Logistic regression"}
RNAME = {"1km": "1 km", "5km": "5 km", "10km": "10 km", "20km": "20 km", "25km": "25 km", "Rinf": "R∞"}
PROV = {"TH50": "Chiang Mai", "TH51": "Lamphun", "TH52": "Lampang", "TH53": "Uttaradit", "TH54": "Phrae", "TH55": "Nan",
        "TH56": "Phayao", "TH57": "Chiang Rai", "TH58": "Mae Hong Son"}


def lab(x):
    a, r = x.split("_")
    return f"{ALG[a]}, {RNAME[r]}"


def f3(v):
    return f"{v:.3f}".replace("-", "−")


def num(v):
    v = int(round(v))
    return f"{v:,}" if v >= 10000 else str(v)


def table(head, rows, align):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


blocks = [MARK, "## Post hoc sensitivity analyses (Section 3.6)",
          "The tables and figure below were specified on 5 October 2026, after the results in the main text had been read, "
          "and computed from saved outputs without refitting any model. They are post hoc and do not amend the pre-declaration."]
# S14 paired AUC differences
s = pd.read_csv(PH / "S1_paired_auc_differences.csv")
rows = []
for _, r in s.iterrows():
    comp = f"{ALG[r.a]} − {ALG[r.b]}" if r.comparison == "algorithm pair" else f"{RNAME[r.a.split('_')[1]]} − 10 km ({ALG[r.a.split('_')[0]]})"
    rows.append([r.comparison.capitalize(), RNAME[r.radius] if r.comparison == "algorithm pair" else "–", comp,
                 f3(r.median_dAUC), f"{f3(r.min_dAUC)} to {f3(r.max_dAUC)}", f"{int(r.folds_a_higher)}/{int(r.folds)}"])
blocks += ["**Table S14.** Paired differences in held-out AUC (post hoc). Each difference is taken within fold and background draw, "
           "summarized per fold as the median over the five draws. The table gives the median and range across the nine folds, and the number of folds "
           "in which the first configuration had the higher AUC. AUC, area under the receiver operating characteristic curve.",
           table(["Comparison", "Radius", "Difference", "Median", "Range across folds", "Folds first higher"], rows, "lllrrr")]
# S15 Jaccard by cut-off
j = pd.read_csv(PH / "S2_jaccard_by_cutoff.csv")
rows = []
for (kind, a, b), g in j.groupby(["comparison", "a", "b"], sort=False):
    g = g.set_index("top_share")
    rows.append([kind.capitalize(), f"{lab(a)} vs {lab(b)}"] +
                [f"{g.loc[p, 'median_over_provinces']:.2f} [{g.loc[p, 'min_province']:.2f}–{g.loc[p, 'max_province']:.2f}]"
                 for p in (0.05, 0.10, 0.20)])
blocks += ["**Table S15.** Hotspot overlap (Jaccard) when hotspots are defined as each map's study-wide top 5%, 10% or 20% "
           "(post hoc). Median of the nine provincial values, each a median over the five background draws, with the provincial "
           "range in brackets. The 10% column reproduces the values reported in the main text.",
           table(["Comparison", "Maps", "Top 5%", "Top 10%", "Top 20%"], rows, "llrrr")]
# S16 consensus by f cut-off
c = pd.read_csv(PH / "S2_consensus_by_f_cutoff.csv")
rows = [["All 90 maps" if r.set.startswith("A") else "75 maps without the 1 km ring", num(r["km2_f_ge_0.8"]),
         num(r["km2_f_ge_0.9"]), num(r["km2_f_eq_1"])] for _, r in c.iterrows()]
blocks += ["**Table S16.** Area (km²) that was a hotspot (study-wide top 10%) in at least 80%, at least 90% and all of the maps "
           "(post hoc for the 80% and 100% cuts, while the 90% cut defines the consensus core).",
           table(["Map set", "f ≥ 0.8", "f ≥ 0.9", "f = 1"], rows, "lrrr")]
# S17 hotspot counts
k = pd.read_csv(PH / "S2_hotspot_counts_10km_algorithm_pairs.csv")
rows = [[f"{ALG[r.a.split('_')[0]]} vs {ALG[r.b.split('_')[0]]}", PROV[r.province], num(r.n_top10_a), num(r.n_top10_b),
         num(r.n_intersection), num(r.n_union), f"{r.jaccard:.2f}"] for _, r in k.iterrows()]
blocks += ["**Table S17.** Hotspot cell counts behind the provincial Jaccard values for the algorithm pairs at the 10 km radius: "
           "cells in the study-wide top 10% of each map within the province (H_A, H_B), their intersection and union, and the "
           "Jaccard index, as medians over the five background draws, so the Jaccard column need not equal the ratio of the medians. "
           "Counts for every comparison and draw are in the released map-similarity table. One cell is 0.25 km².",
           table(["Algorithm pair", "Province", "Cells in H_A", "Cells in H_B", "Intersection", "Union", "Jaccard"], rows, "llrrrrr")]
# S18 interaction
d = pd.read_csv(PH / "S3_radius_x_algorithm.csv")
rows = [["All 90 maps" if r.set.startswith("A") else "75 maps without the 1 km ring", num(r.dependent_cells),
         f"{r.share_radius:.2f}", f"{r.share_algorithm:.2f}", f"{r.share_draw:.2f}", f"{r.share_radius_x_algorithm:.2f}",
         f"{r.share_interactions_with_draw:.2f}"] for _, r in d.iterrows()]
blocks += ["**Table S18.** Shares of variation in top-10% hotspot membership over the configuration-dependent cells, with the "
           "radius × algorithm interaction separated from the interactions involving the background draw (post hoc). Shares are "
           "rounded and may not sum exactly to 1.",
           table(["Map set", "Cells", "Radius", "Algorithm", "Draw", "Radius × algorithm", "Interactions with draw"], rows, "lrrrrrr")]
# S19 distances + Figure S1
t = pd.read_csv(PH / "S4_train_test_distances.csv")
rows = [[f"{PROV[r.fold]} ({r.fold})", r.n_test_presences, r.n_test_background,
         f"{r.presence_to_train_km_median:.1f} ({r.presence_to_train_km_p10:.1f}–{r.presence_to_train_km_p90:.1f})",
         f"{r.background_to_train_km_median:.1f} ({r.background_to_train_km_p10:.1f}–{r.background_to_train_km_p90:.1f})"]
        for _, r in t.iterrows()]
blocks += ["**Table S19.** Distance from the test points of each fold (held-out province) to the nearest training presence "
           "(presences of the other eight provinces): median, with the 10th and 90th percentiles in parentheses, in km (post hoc). "
           "The folds are mapped in Figure S1.",
           table(["Fold (held-out province)", "Test presences", "Test background", "Test presences to training (km)",
                  "Test background to training (km)"], rows, "lrrrr")]
blocks += ["![](../06_posthoc_sensitivity/figS1_fold_map.png) {#nonum}",
           "**Figure S1.** The nine leave-one-province-out folds: province code, name and number of presences held out in each "
           "fold (post hoc). Firm locations are not shown. WGS 84 / UTM zone 47N (EPSG:32647). Boundaries from the OCHA "
           "subnational boundaries dataset cited in the main text."]
m = pd.read_csv(PH / "S4_registry_polygon_mismatches.csv")
rows = [[r.record, r.registry_province, r.lies_in_polygon_of, r.distance_to_polygon_edge.replace(",", "")
         if int(r.distance_to_polygon_edge.replace(",", "").split()[0]) < 10000 else r.distance_to_polygon_edge]
        for _, r in m.iterrows()]
blocks += ["**Table S20.** The six presences whose registry province differs from the provincial polygon they lie in. They are "
           "kept under their registry province, so in these folds the holdout is not strictly geographic. Registry identifiers are "
           "withheld.",
           table(["Record", "Registry province (fold)", "Lies in polygon of", "Distance to that polygon's edge"], rows, "rllr")]

text = SUP.read_text(encoding="utf-8")
text = text.split(MARK)[0].rstrip() + "\n\n" + "\n\n".join(blocks) + "\n"
SUP.write_text(text, encoding="utf-8")
print("appended S14-S20 + Figure S1")
