"""Append the post hoc reviewer-requested tables S21-S26 to supplementary_src.md (idempotent: an earlier appended block is
replaced). Run after make_supplementary.py and make_supplementary_posthoc.py, which rewrite everything before this block.
Source: Paper_D/08_revision_posthoc/outputs/ (spec hashed 2026-11-01 before computing, Note 1 appended)."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RV = HERE.parent / "08_revision_posthoc" / "outputs"
TP = HERE.parents[1] / "exp_map_stability" / "02_data" / "test_points"
SUP = HERE / "supplementary_src.md"
MARK = "<!-- REVISION-R1-POSTHOC-START -->"
ALG = {"RF": "Random forest", "XGB": "XGBoost", "LR": "Logistic regression"}
ORDER = ["RF", "XGB", "LR"]
RAD = ["1km", "5km", "10km", "20km", "25km", "Rinf"]
RNAME = {"1km": "1 km", "5km": "5 km", "10km": "10 km", "20km": "20 km", "25km": "25 km", "Rinf": "R∞"}
PROV = {"TH50": "Chiang Mai", "TH51": "Lamphun", "TH52": "Lampang", "TH53": "Uttaradit", "TH54": "Phrae", "TH55": "Nan",
        "TH56": "Phayao", "TH57": "Chiang Rai", "TH58": "Mae Hong Son"}
FEAT = {"slope_deg": "Slope", "dist_road_km": "Distance to road", "dist_rail_km": "Distance to existing railway",
        "dist_water_km": "Distance to water", "dist_cbd_km": "Distance to provincial capital",
        "dist_dryport_km": "Distance to dry port", "nightlight": "Nighttime light", "forest": "Forest",
        "builtup": "Built-up", "dist_railway_new_km": "Distance to planned railway",
        "dist_border_CK_km": "Distance to Chiang Khong crossing"}


def neg(s):
    return s.replace("-", "−")


def f3(v):
    return neg(f"{v:.3f}")


def sg(v, d=3):
    return neg(f"{v:+.{d}f}")


def table(head, rows, align):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def lab(x):
    a, r = x.split("_")
    return f"{ALG[a]}, {RNAME[r]}"


blocks = [MARK, "## Post hoc reviewer-requested sensitivity analyses (Sections 2.13, 3.1–3.3 and 3.7)",
          "The tables below were specified on 1 November 2026 in response to peer review, after the results in the main text "
          "had been read. The specification was hashed before any of them was computed. They are post hoc and do not amend "
          "the pre-declaration. Neighboring cells are spatially autocorrelated, so no significance test is applied."]

# S21 near-facility test background (P1)
u = pd.read_csv(RV / "P1_auc_by_unit.csv")
COLS = [("auc_original", "Province-wide (pre-declared)"), ("auc_near5km", "0.5–5 km from a presence"),
        ("auc_near1km", "0.5–1 km from a presence")]
rows = []
for L in ORDER:
    for r in RAD:
        g = u[(u.learner == L) & (u.radius == r)]
        cells = []
        for c, _ in COLS:
            summ = g.groupby("seed")[c].median().median()
            prov = g.groupby("fold")[c].median()
            cells.append(f"{summ:.3f} ({prov.min():.3f}–{prov.max():.3f})")
        rows.append([ALG[L], RNAME[r]] + cells)
assert u.reproduced.all() and len(u) == 810
orig = pd.concat([pd.read_csv(p) for p in sorted(TP.glob("test_points_TH*.csv"))])
comp = []
for name, df in [("province-wide", orig), ("0.5–5 km", pd.read_csv(RV / "P1_test_points_near5km.csv")),
                 ("0.5–1 km", pd.read_csv(RV / "P1_test_points_near1km.csv"))]:
    assert len(df) == 2859
    comp.append(f"{name} {100 * df.forest.mean():.1f}% and {100 * df.builtup.mean():.1f}%")
blocks += ["**Table S21.** Held-out AUC against three shared test backgrounds (post hoc, reviewer-requested): the pre-declared "
           "background drawn across each held-out province, and backgrounds drawn in the same way but restricted to cells "
           "0.5–5 km or 0.5–1 km from the nearest presence, with the same number of points per province. Median over the five "
           "background draws of the median across the nine folds, with the range across folds (each a median over draws) in "
           "parentheses. The 810 refitted fold-level models reproduced the stored AUC on the pre-declared test sets (maximum "
           "absolute difference 1.1 × 10⁻¹⁶). Share of test background points on forest and on built-up cells: "
           + ", ".join(comp) + ". The test backgrounds are sampled points, not land confirmed to be developable. AUC, area "
           "under the receiver operating characteristic curve.",
           table(["Algorithm", "Radius"] + [h for _, h in COLS], rows, "llrrr")]
# S21 continued: by province (author request 2026-11-03), from the same P1 outputs, one value per cell
ntest = pd.read_csv(RV / "P1_test_points_near1km_manifest.csv").set_index("ADM1_PCODE")["n_test"]
med = u.groupby(["fold", "radius", "learner"])[[c for c, _ in COLS]].median()
SHORT = {"RF": "RF", "XGB": "XGBoost", "LR": "LR"}
rows_a, rows_b, signs = [], [], {c: 0 for c, _ in COLS}
for f in PROV:
    for L in ORDER:
        lvl, dif, rng = [], [], []
        g = u[(u.fold == f) & (u.radius == "10km") & (u.learner == L)]
        for c, _ in COLS:
            a1, a10 = med.loc[(f, "1km", L), c], med.loc[(f, "10km", L), c]
            signs[c] += (a1 < a10) if c != "auc_near1km" else (a1 > a10)
            lvl.append(f"{a10:.3f}")
            dif.append(sg(a1 - a10))
            rng.append(f"{g[c].max() - g[c].min():.3f}")
        rows_a.append([PROV[f], int(ntest[f]), SHORT[L]] + lvl + dif)
        rows_b.append([PROV[f], int(ntest[f]), SHORT[L]] + rng)
assert signs == {"auc_original": 27, "auc_near5km": 26, "auc_near1km": 25}, signs
TB = ["Province-wide", "0.5–5 km", "0.5–1 km"]
blocks += ["**Table S21 (continued).** Held-out AUC by province (post hoc, reviewer-requested): AUC of the 10 km ring and the "
           "difference between the 1 km and the 10 km ring (1 km minus 10 km, positive when the 1 km ring scored higher), for "
           "each test background, each AUC the median over the five background draws. n is the number of test presences, equal "
           "to the number of test background points, and is the same in all three test backgrounds. All, the province-wide test "
           "background, 0.5–5 and 0.5–1, the test backgrounds 0.5–5 km and 0.5–1 km from the nearest presence, Δ, AUC of the "
           "1 km ring minus AUC of the 10 km ring. RF, random forest, LR, logistic regression.",
           table(["Province", "n", "Algorithm", "All", "0.5–5", "0.5–1", "Δ all", "Δ 0.5–5", "Δ 0.5–1"],
                 rows_a, "lrlrrrrrr"),
           "**Table S21 (continued).** Range of the held-out AUC of the 10 km ring across the five background draws, by province "
           "and test background (post hoc, reviewer-requested). The draws share one test set per province, so the range "
           "describes the stability of the fitted models, not the sampling precision of the test set, which is lower in "
           "provinces with few test points (n as above).",
           table(["Held-out province", "n", "Algorithm"] + TB, rows_b, "lrlrrr")]

# S22 between-draw agreement at 10 km (P4)
b = pd.read_csv(RV / "P4_between_draw_10km_summary.csv").set_index("learner")
rows = [[ALG[L]] + [f"{b.loc[L, m + '_median']:.3f} [{b.loc[L, m + '_min']:.3f}–{b.loc[L, m + '_max']:.3f}]"
                    for m in ("ssim_raw_7", "ssim_rank_7", "jaccard_top10")] for L in ORDER]
blocks += ["**Table S22.** Agreement between the five background draws of the same algorithm at the 10 km radius (post hoc, "
           "reviewer-requested): median over the 10 draw pairs in each province, then the median of the nine provincial values "
           "with their range in brackets. SSIM uses the 7 × 7 window. This is the baseline for the between-algorithm values at "
           "10 km in Table S6, which compare algorithms within the same draw.",
           table(["Algorithm", "SSIM raw", "SSIM rank", "Jaccard top 10%"], rows, "lrrr")]

# S23 structure term (P3)
s = pd.read_csv(RV / "P3_ssim_structure_summary.csv")
rows = []
for comp_, sub in [("across_radii", "Radius against 10 km"), ("across_learners", "Algorithm pair")]:
    g = s[s.comparison == comp_].copy()
    g["r"] = g.a.str.split("_").str[1].map(RAD.index)
    g["l"] = g.a.str.split("_").str[0].map(ORDER.index)
    g = g.sort_values(["l", "r"] if comp_ == "across_radii" else ["r", "l"])
    for _, r in g.iterrows():
        rows.append([sub, f"{lab(r.a)} vs {lab(r.b)}",
                     f"{f3(r.ssim_rank_7_full_median)} [{f3(r.ssim_rank_7_full_min)}–{f3(r.ssim_rank_7_full_max)}]",
                     f"{f3(r.ssim_rank_7_structure_median)} [{f3(r.ssim_rank_7_structure_min)}–"
                     f"{f3(r.ssim_rank_7_structure_max)}]"])
blocks += ["**Table S23.** Rank SSIM and its structure term (post hoc, reviewer-requested), on the percentile-rank maps with the "
           "7 × 7 window. The structure term is t_str = (σ_ab + C3)/(σ_a σ_b + C3), C3 = C2/2, and compares local pattern without "
           "the local mean and variance. Median over the five draws in each province, then the median of the nine provincial "
           "values with their range in brackets.",
           table(["Comparison", "Maps", "Rank SSIM", "Structure term"], rows, "llrr")]

# S24 SHAP across draws (P5)
p = pd.read_csv(RV / "P5_shap_all_draws.csv")
assert len(p) == 90 * 11
rows = []
for L in ORDER:
    g = p[p.learner == L]
    agg = g.groupby("feature").agg(ma=("mean_abs_shap", "median"), ma0=("mean_abs_shap", "min"),
                                   ma1=("mean_abs_shap", "max"), sh=("share", "median"), sh0=("share", "min"),
                                   sh1=("share", "max"), top3=("rank", lambda x: int((x <= 3).sum())))
    for f, r in agg.sort_values("sh", ascending=False).iterrows():
        rows.append([ALG[L], FEAT[f], f"{r.ma:.3f} ({r.ma0:.3f}–{r.ma1:.3f})", f"{r.sh:.2f} ({r.sh0:.2f}–{r.sh1:.2f})",
                     f"{r.top3}/30"])
blocks += ["**Table S24.** SHAP across the five background draws (post hoc, reviewer-requested). Mean absolute SHAP value on "
           "the probability scale and normalized share (mean absolute SHAP value of the covariate divided by the sum over the "
           "eleven covariates), as the median over the six radii and five draws with the range over these 30 map models in "
           "parentheses, and the number of the 30 map models in which the covariate ranks in the top three. The 72 refitted map "
           "models of draws 0, 1, 7 and 2024 reproduced the saved maps exactly. Logistic-regression values are sampled "
           "approximations. Directions and the group-removal test were not repeated.",
           table(["Algorithm", "Covariate", "Mean absolute SHAP", "Normalized share", "Top three"], rows, "llrrr")]
rows = []
for L in ORDER:
    for r in RAD:
        g = p[(p.learner == L) & (p.radius == r)]
        cells = []
        for f in ("slope_deg", "dist_road_km"):
            h = g[g.feature == f]
            cells += [f"{int(h['rank'].min())}–{int(h['rank'].max())}" if h['rank'].nunique() > 1 else str(int(h['rank'].iloc[0])),
                      f"{h.share.median():.2f} ({h.share.min():.2f}–{h.share.max():.2f})"]
        rows.append([ALG[L], RNAME[r]] + cells)
blocks += ["**Table S24 (continued).** Rank among the eleven covariates and normalized share of slope and distance to road in "
           "each configuration across the five background draws: range of ranks, and median share with its range over the "
           "five draws in parentheses.",
           table(["Algorithm", "Radius", "Slope rank", "Slope share", "Road rank", "Road share"], rows, "llrrrr")]

# S25 province assignment (P2)
a = pd.read_csv(RV / "P2_auc_summary.csv")
rows = []
for L in ORDER:
    for r in RAD:
        x = a[(a.learner == L) & (a.radius == r)].set_index("variant")
        rows.append([ALG[L], RNAME[r], f"{x.loc['reassign', 'auc_submitted']:.3f}",
                     f"{x.loc['reassign', 'auc']:.3f} ({sg(x.loc['reassign', 'd_auc'])})",
                     f"{x.loc['exclude', 'auc']:.3f} ({sg(x.loc['exclude', 'd_auc'])})"])
blocks += ["**Table S25.** Held-out AUC with the six presences of Table S20 reassigned to the province whose polygon contains "
           "them or excluded (post hoc, reviewer-requested). All 270 units were rerun for each variant (810 fits each), with "
           "the same seeds and sampling rules, and scored on the original test backgrounds. Median over the five draws of the "
           "median across the nine folds, with the change from the reported value in parentheses. Because the training "
           "presences change, the training backgrounds are regenerated, so unit-level differences reflect both the six records "
           "and the regenerated backgrounds.",
           table(["Algorithm", "Radius", "Reported", "Reassigned", "Excluded"], rows, "llrrr")]
d = pd.read_csv(RV / "P2_delta_by_province.csv")
rows = []
for f in PROV:
    x = d[d.fold == f].set_index("variant")
    rows.append([f"{PROV[f]} ({f})"] + [f"{int(x.loc[v, 'n_test_presence'])}" for v in ("reassign", "exclude")] +
                [f"{sg(x.loc[v, 'median_d_auc'], 4)} ({sg(x.loc[v, 'min_d_auc'])} to {sg(x.loc[v, 'max_d_auc'])})"
                 for v in ("reassign", "exclude")])
blocks += ["**Table S25 (continued).** Change in held-out AUC by fold: test presences in each variant, and the median change "
           "over the 90 fold-level fits of the fold (6 radii × 5 draws × 3 algorithms) with its range in parentheses.",
           table(["Fold (held-out province)", "Test presences, reassigned", "Test presences, excluded",
                  "ΔAUC, reassigned", "ΔAUC, excluded"], rows, "lrrrr")]

# S26 descriptive variance decomposition (P6)
v = pd.read_csv(RV / "P6_variance_components.csv")
TERMS = [("R", "Radius", "vc_share_R"), ("L", "Algorithm", "vc_share_L"), ("RL", "Radius × algorithm", "vc_share_RL"),
         ("D", "Background draw", "vc_share_D"), ("RD", "Radius × draw", "vc_share_RD"),
         ("LD", "Algorithm × draw", "vc_share_LD"), ("RLD", "Radius × algorithm × draw (residual)", "vc_share_res_RLD")]
rows = []
for st, lab_ in [("A_all90", "All 90 maps"), ("B_no1km_75", "75 maps without the 1 km ring")]:
    x = v[(v.set == st) & (v.province == "ALL")].iloc[0]
    pr = v[(v.set == st) & (v.province != "ALL")]
    for t, name, vc in TERMS:
        rows.append([lab_, name, int(x[f"df_{t}"]), f"{x[f'ss_share_{t}']:.2f}", f"{x[f'MS_{t}']:.3f}", f"{x[vc]:.2f}",
                     f"{pr[vc].min():.2f}–{pr[vc].max():.2f}"])
blocks += ["**Table S26.** Descriptive variance decomposition of top-10% hotspot membership over the configuration-dependent "
           "cells (post hoc, reviewer-requested). Degrees of freedom per cell, raw share of the sum of squares, mean square "
           "(pooled over cells) and variance-component share from expected mean squares (method-of-moments, ANOVA "
           "estimator), with radius and algorithm fixed, "
           "the background draw random and the three-way interaction as the residual, negative estimates set to zero (none occurred). The last "
           "column gives the range of the variance-component share over the nine provinces. Cells are spatially "
           "autocorrelated, so no test or interval is attached. Shares are rounded and may not sum exactly to 1.",
           table(["Map set", "Term", "df", "SS share", "Mean square", "Component share", "Range over provinces"], rows,
                 "llrrrrr")]

text = SUP.read_text(encoding="utf-8")
text = text.split(MARK)[0].rstrip() + "\n\n" + "\n\n".join(blocks) + "\n"
for blk in blocks:
    assert "—" not in blk, blk[:60]
SUP.write_text(text, encoding="utf-8")
print("appended S21-S26")
