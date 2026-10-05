"""Manuscript figures (read-only inputs; outputs only in this folder).

Fig 1 study area (new) · Fig 2 method flowchart (new) · Fig 3 discrimination by radius (new: bars = range across
the nine provinces, not across background draws) · Fig 4 10 km maps, Fig 5 provincial Jaccard, Fig 6 factor
reliance (experiment figures, footnotes trimmed; their content moves to the captions) · Fig 7 consensus (redrawn).
Palette = the experiment's validated learner palette (validate_palette.js: all checks PASS, green contrast WARN ->
direct labels + Table 4 as the table view).
"""
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
RFRP = HERE.parents[1]
EXP = RFRP / "exp_map_stability"
POST = HERE.parent / "01_posthoc_consensus"
BND = RFRP / "paper_B" / "03_results" / "rasters" / "boundary_9prov_32647.gpkg"
PRES = RFRP / "data" / "processed" / "analysis_dataset_m3_base_clean.csv"

SURFACE, INK, INK2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COL = {"RF": "#2a78d6", "XGB": "#eb6834", "LR": "#1baf7a"}
NAME = {"RF": "Random forest", "XGB": "XGBoost", "LR": "Logistic regression"}
plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False, "text.color": INK})
DPI = 600


def num(x):
    """MDPI: thousands comma only for five or more digits."""
    return f"{x:,.0f}" if abs(x) >= 10000 else f"{x:.0f}"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def scalebar(ax, x0, y0, km=50):
    ax.plot([x0, x0 + km * 1000], [y0, y0], color=INK, lw=2, solid_capstyle="butt", zorder=5)
    ax.text(x0 + km * 500, y0 + 6000, f"{km} km", ha="center", va="bottom", fontsize=7, color=INK2)


def north(ax, x, y):
    ax.annotate("N", xy=(x, y), xytext=(x, y - 30000), ha="center", va="center", fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))


bnd = gpd.read_file(BND)

# ------------------------------------------------------------------ Fig 1 study area
# (a) Thailand locator + (b) study area: own script since 2026-10-05
import runpy
runpy.run_path(str(HERE / "make_fig_study_area.py"))

# ------------------------------------------------------------------ Fig 2 method flowchart
fig, ax = plt.subplots(figsize=(6.5, 8.6))
ax.set_xlim(0, 100); ax.set_ylim(4, 128); ax.set_axis_off()
FILL = {"lock": "#e8f1fc", "core": "#cde2fb", "q": "#f4f3ee", "syn": "#ffffff", "use": "#f4f3ee"}


def box(x, y, w, h, title, body, kind, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=FILL[kind],
                                ec="#256abf" if kind != "q" else INK2, lw=1.0, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.2, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=6.8, color=INK2, linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))


box(12, 113, 76, 13, "Step 1  Lock data and design",
    "2859 presences · 11 covariates · 9 provinces · 500 m grid\n"
    "Pre-declaration, inputs, test points and code hashed before any fit", "lock")
arrow(50, 112.4, 50, 108.6)
box(12, 95, 76, 13, "Step 2  Shared held-out test sets (Section 2.6)",
    "Leave-one-province-out, 9 folds\n"
    "Test background ≥ 0.5 km from all presences, n = province presences (Equation (1))\n"
    "The same test points for every configuration", "lock")
arrow(50, 94.4, 50, 90.6)
box(12, 74, 76, 16.5, "Step 3  Crossed experiment (Section 2.7)",
    "Modeling choices: 6 background radii (1, 5, 10, 20, 25 km, R∞; Equation (2))\n"
    "× 3 algorithms (random forest, XGBoost, logistic regression) = 18 configurations\n"
    "Experimental setting: 5 background draws per configuration, summarized by median\n"
    "270 LOPO units · 90 maps · fixed settings · no per-map rescaling", "core")
for xc in (17, 50, 83):
    arrow(50, 73.4, xc, 65.6)
box(2, 44, 30, 20.5, "Step 4a  RQ1\nDiscrimination",
    "\nHeld-out AUC (Equation (4))\nTSS at a training-only\nthreshold (Equations (5) and (6))\nκ ≡ TSS (Equations (7) and (8))", "q")
box(35, 44, 30, 20.5, "Step 4b  RQ2\nMap agreement",
    "\nSSIM, raw and rank,\n7 × 7 window (Equations (9)–(11))\nTop-10% hotspot Jaccard\n(Equations (12) and (13))",
    "q")
box(68, 44, 30, 20.5, "Step 4c  RQ3\nFactor reliance",
    "\nMean |SHAP| and direction\n(Equation (14))\nPermutation ΔAUC (Equation (15))\nLeave-one-group-out\n(Equation (16))", "q")
for xc in (17, 50, 83):
    arrow(xc, 43.4, 50, 37.6)
box(12, 21, 76, 16.5, "Step 5  Synthesis (exploratory, Section 2.11)",
    "Hotspot frequency f over 90 maps (Equation (17)), no reference ring\n"
    "Consensus core (f ≥ 0.9) · configuration-dependent (0 < f < 0.9) · never a hotspot\n"
    "Share of variation: radius · algorithm · remainder (Equation (18))", "syn", dashed=True)
arrow(50, 20.4, 50, 16.6)
box(12, 5, 76, 11.5, "Step 6  Limits of use (Section 2.12)",
    "Regional screening, not parcel selection\nNo configuration selected on AUC alone · no significance test on pixels",
    "use")
save(fig, "fig_method_flowchart")

# ------------------------------------------------------------------ Fig 3 discrimination by radius
fold = pd.read_csv(EXP / "02_data" / "runs" / "lopo_metrics_by_fold.csv")
RAD = ["1km", "5km", "10km", "20km", "25km", "Rinf"]
XL = ["1", "5", "10", "20", "25", "R∞"]
cen = fold.groupby(["learner", "radius", "seed"])[["auc", "tss"]].median().groupby(["learner", "radius"]).median()
prov = fold.groupby(["learner", "radius", "fold"])[["auc", "tss"]].median()
lo, hi = prov.groupby(["learner", "radius"]).min(), prov.groupby(["learner", "radius"]).max()
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, m, lab, panel in zip(axs, ("auc", "tss"), ("AUC on held-out province", "TSS on held-out province"),
                             ("a", "b")):
    for k, L in enumerate(("RF", "XGB", "LR")):
        x = np.arange(6) + (k - 1) * 0.16
        y = [cen.loc[(L, r), m] for r in RAD]
        e = np.array([[y[i] - lo.loc[(L, r), m], hi.loc[(L, r), m] - y[i]] for i, r in enumerate(RAD)]).T
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=COL[L], elinewidth=1.0, alpha=0.45, zorder=2)
        ax.plot(x, y, "-o", color=COL[L], lw=1.6, ms=4.2, mec="white", mew=0.8, zorder=3, label=NAME[L])
    ax.set_xticks(range(6), XL); ax.set_xlim(-0.5, 5.5)
    ax.set_xlabel("Outer radius of the training-background ring (km)"); ax.set_ylabel(lab)
    ax.set_title(panel, loc="center", fontsize=9, fontweight="bold", pad=4); ax.grid(axis="x", visible=False)
axs[0].legend(loc="lower right", fontsize=7)
fig.tight_layout()
save(fig, "fig_discrimination")


# ------------------------------------------------------------------ Fig 7 consensus (redrawn, no suptitle)
valid = np.load(EXP / "02_data" / "maps" / "valid_mask.npy")
H, W = valid.shape
minx, miny, maxx, maxy = bnd.union_all().bounds
minx, maxy = np.floor(minx / 500) * 500, np.ceil(maxy / 500) * 500
ext = (minx, minx + W * 500, maxy - H * 500, maxy)


def grid(v):
    g = np.full((H, W), np.nan, np.float32); g[valid] = v; return g


fA, fB = np.load(POST / "consensus_freq_setA.npy"), np.load(POST / "consensus_freq_setB.npy")
seq = LinearSegmentedColormap.from_list("blue", ["#f2f1ec", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
cls_cmap = ListedColormap(["#ebeae4", "#f0a830", "#184f95"])
fig = plt.figure(figsize=(7.2, 3.4))
axs = [fig.add_axes([0.005 + i * 0.333, 0.20, 0.325, 0.70]) for i in range(3)]
im = axs[0].imshow(grid(fA), cmap=seq, vmin=0, vmax=1, extent=ext, interpolation="nearest")
axs[0].set_title("a", loc="center", fontsize=9, fontweight="bold", pad=4)
for ax, f, t in ((axs[1], fA, "b"), (axs[2], fB, "c")):
    c = np.where(f >= 0.9, 2, np.where(f > 0, 1, 0)).astype(np.float32)
    ax.imshow(grid(c), cmap=cls_cmap, vmin=0, vmax=2, extent=ext, interpolation="nearest")
    ax.set_title(t, loc="center", fontsize=9, fontweight="bold", pad=4)
for ax in axs:
    bnd.boundary.plot(ax=ax, color=INK2, linewidth=0.35)
    ax.set_axis_off(); ax.set_aspect("equal")
scalebar(axs[0], ext[0] + 5000, ext[2] + 10000)
north(axs[0], ext[1] - 15000, ext[3] - 5000)
cax = fig.add_axes([0.04, 0.12, 0.26, 0.03])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("f, share of maps with the cell in the top 10%", fontsize=6.5, color=INK2)
cb.ax.tick_params(labelsize=6)
fig.legend([Patch(color=cls_cmap(i)) for i in range(3)],
           ["Never a hotspot (f = 0)", "Configuration-dependent (0 < f < 0.9)", "Consensus core (f ≥ 0.9)"],
           loc="lower left", ncol=1, fontsize=7, bbox_to_anchor=(0.36, 0.0))
save(fig, "fig_consensus")

# ------------------------------------------------------------------ map agreement summary (covers former Tables 5-6)
sim = pd.read_csv(EXP / "02_data" / "runs" / "map_similarity_summary_across_seeds.csv")
PAIRS = [("RF", "XGB"), ("RF", "LR"), ("XGB", "LR")]
PCOL = {("RF", "XGB"): "#4a3aa7", ("RF", "LR"): "#e87ba4", ("XGB", "LR"): "#eda100"}   # validated, all-pairs PASS
PMK = {("RF", "XGB"): "o", ("RF", "LR"): "s", ("XGB", "LR"): "D"}
LMK = {"RF": "o", "XGB": "s", "LR": "D"}
SHORT = {"RF": "RF", "XGB": "XGBoost", "LR": "LR"}
MET = [("ssim_raw_7", "SSIM, raw scores"), ("ssim_rank_7", "SSIM, percentile ranks"),
       ("jaccard_top10", "Jaccard, top-10% hotspots")]


def provvals(kind, within, a, b, m):
    s = sim[(sim.comparison == kind) & (sim.within == within) & (sim.a == a) & (sim.b == b) & (sim.metric == m)]
    v = s["median"].to_numpy()
    assert len(v) == 9
    return np.median(v), v.min(), v.max()


fig, axs = plt.subplots(3, 2, figsize=(7.0, 7.4), sharey="row")
RNO = [r for r in RAD if r != "10km"]
for i, (m, mlab) in enumerate(MET):
    ax = axs[i, 0]
    for k, L in enumerate(("RF", "XGB", "LR")):
        x = np.arange(len(RNO)) + (k - 1) * 0.15
        st = [provvals("across_radii", L, f"{L}_{r}", f"{L}_10km", m) for r in RNO]
        ax.vlines(x, [s[1] for s in st], [s[2] for s in st], color=COL[L], lw=1.0, alpha=0.45)
        ax.plot(x, [s[0] for s in st], "-", color=COL[L], lw=1.4, marker=LMK[L], ms=4, mec="white", mew=0.7,
                label=NAME[L])
    ax.set_xticks(range(len(RNO)), [XL[RAD.index(r)] for r in RNO])
    ax.set_ylabel(mlab)
    ax = axs[i, 1]
    for k, (a, b) in enumerate(PAIRS):
        x = np.arange(6) + (k - 1) * 0.15
        st = [provvals("across_learners", r, f"{a}_{r}", f"{b}_{r}", m) for r in RAD]
        ax.vlines(x, [s[1] for s in st], [s[2] for s in st], color=PCOL[(a, b)], lw=1.0, alpha=0.5)
        ax.plot(x, [s[0] for s in st], "-", color=PCOL[(a, b)], lw=1.4, marker=PMK[(a, b)], ms=4, mec="white",
                mew=0.7, label=f"{SHORT[a]} vs {SHORT[b]}")
    ax.set_xticks(range(6), XL)
    for ax in axs[i]:
        ax.set_ylim(-0.3 if m != "jaccard_top10" else 0, 1.02)
        ax.grid(axis="x", visible=False)
axs[0, 0].set_title("a", loc="center", fontsize=9, fontweight="bold", pad=4)
axs[0, 1].set_title("b", loc="center", fontsize=9, fontweight="bold", pad=4)
for ax in axs[2]:
    ax.set_xlabel("Outer radius of the training-background ring (km)")
axs[2, 0].legend(loc="lower right", fontsize=6.5)
axs[2, 1].legend(loc="lower right", fontsize=6.5)
fig.tight_layout()
save(fig, "fig_agreement_summary")

# ------------------------------------------------------------------ permutation and leave-one-group-out by ring
import json
rows = []
for p in (EXP / "02_data" / "runs" / "lopo").glob("*.json"):
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("status") != "complete":
        continue
    for L, v in d["learners"].items():
        for k_, (m_, _sd) in v["perm_auc_drop"].items():
            rows.append(("perm", d["radius"], k_, m_))
        for g, x_ in v.get("lofo_auc_drop", {}).items():
            rows.append(("lofo", d["radius"], g, x_))
pr = pd.DataFrame(rows, columns=["kind", "radius", "name", "drop"])
assert (pr[pr.kind == "perm"].groupby(["name", "radius"]).size() == 135).all()
assert (pr[pr.kind == "lofo"].groupby(["name", "radius"]).size() == 27).all()
PF = [("builtup+nightlight", "Built-up + nighttime light (joint)", "#0b0b0b", "-"),
      ("builtup", "Built-up", "#2a78d6", "-"), ("slope_deg", "Slope", "#eb6834", "-"),
      ("dist_road_km", "Distance to road", "#1baf7a", "-"), ("nightlight", "Nighttime light", "#4a3aa7", "--"),
      ("forest", "Forest", "#e87ba4", "--")]
LG = [("physical_capacity", "Physical capacity", "#eb6834"), ("development_intensity", "Development intensity", "#2a78d6"),
      ("transport_access", "Transport access", "#1baf7a"), ("forest_area_condition", "Forest condition", "#e87ba4"),
      ("market_gateway_access", "Market and gateway access", "#898781")]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
for key, lab, c, ls in PF:
    s = pr[(pr.kind == "perm") & (pr.name == key)].groupby("radius")["drop"].median()
    axs[0].plot(range(6), [s[r] for r in RAD], ls=ls, color=c, lw=1.5, marker="o", ms=3.5, mec="white", mew=0.6,
                label=lab)
for key, lab, c in LG:
    s = pr[(pr.kind == "lofo") & (pr.name == key)].groupby("radius")["drop"].median()
    axs[1].plot(range(6), [s[r] for r in RAD], color=c, lw=1.5, marker="o", ms=3.5, mec="white", mew=0.6, label=lab)
axs[0].set_title("a", loc="center", fontsize=9, fontweight="bold", pad=4)
axs[1].set_title("b", loc="center", fontsize=9, fontweight="bold", pad=4)
for ax in axs:
    ax.set_xticks(range(6), XL)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Outer radius of the training-background ring (km)")
    ax.legend(fontsize=6.3, loc="upper right")
axs[0].set_ylabel("Median loss of held-out AUC")
fig.tight_layout()
save(fig, "fig_perm_by_radius")

# ------------------------------------------------------------------ redrawn from data (2026-10-04): no titles or
# subtitles in any figure; panels carry only a centred letter. Covariate names follow Table 1 of the manuscript.
def plab(ax, s):
    ax.set_title(s, loc="center", fontsize=9, fontweight="bold", pad=4, color=INK)


BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
FEATN = {"slope_deg": "Slope", "dist_road_km": "Distance to road", "dist_rail_km": "Distance to existing railway",
         "dist_water_km": "Distance to water", "dist_cbd_km": "Distance to provincial capital",
         "dist_dryport_km": "Distance to dry port", "nightlight": "Nighttime light", "forest": "Forest",
         "builtup": "Built-up", "dist_railway_new_km": "Distance to planned railway",
         "dist_border_CK_km": "Distance to Chiang Khong crossing",
         "builtup+nightlight": "Built-up + nighttime light (joint)"}
PCODE = ["TH50", "TH51", "TH52", "TH53", "TH54", "TH55", "TH56", "TH57", "TH58"]
PNAME = ["Chiang Mai", "Lamphun", "Lampang", "Uttaradit", "Phrae", "Nan", "Phayao", "Chiang Rai", "Mae Hong Son"]
MAPS = EXP / "02_data" / "maps"


def ink_on(v, vmax):
    return "white" if v / vmax > 0.55 else INK


# --- 10 km maps, three algorithms
fig = plt.figure(figsize=(7.2, 3.3))
axs = [fig.add_axes([0.005 + i * 0.333, 0.20, 0.325, 0.72]) for i in range(3)]
for ax, L, s in zip(axs, ("RF", "XGB", "LR"), "abc"):
    v = np.load(MAPS / f"{L}_10km_s42.npy")
    im = ax.imshow(grid(v), cmap=BLUE, vmin=0, vmax=1, extent=ext, interpolation="nearest")
    bnd.boundary.plot(ax=ax, color=INK2, linewidth=0.35)
    ax.set_axis_off(); ax.set_aspect("equal"); plab(ax, s)
scalebar(axs[0], ext[0] + 5000, ext[2] + 10000)
north(axs[0], ext[1] - 15000, ext[3] - 5000)
cax = fig.add_axes([0.30, 0.11, 0.40, 0.03])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("Predicted suitability score (0–1), same scale in every panel", fontsize=7, color=INK2)
cb.ax.tick_params(labelsize=6.5)
save(fig, "fig_maps_10km")

# --- provincial Jaccard heatmaps
RNO = [r for r in RAD if r != "10km"]
rowsA = [(f"{SHORT[L]}: {XL[RAD.index(r)]}" + ("" if r == "Rinf" else " km"), "across_radii", L, f"{L}_{r}", f"{L}_10km")
         for L in ("RF", "XGB", "LR") for r in RNO]
rowsB = [(f"{XL[RAD.index(r)]}" + ("" if r == "Rinf" else " km") + f": {SHORT[a]}–{SHORT[b]}", "across_learners", r,
          f"{a}_{r}", f"{b}_{r}") for r in RAD for a, b in PAIRS]


def jmat(rows):
    M = np.zeros((len(rows), 9))
    for i, (_, kind, within, a, b) in enumerate(rows):
        s = sim[(sim.comparison == kind) & (sim.within == within) & (sim.a == a) & (sim.b == b)
                & (sim.metric == "jaccard_top10")].set_index("province")["median"]
        M[i] = [s[c] for c in PCODE]
    return M


fig, axs = plt.subplots(1, 2, figsize=(7.2, 7.6), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.55})
for ax, rows, step, s in ((axs[0], rowsA, 5, "a"), (axs[1], rowsB, 3, "b")):
    M = jmat(rows)
    im = ax.imshow(M, cmap=BLUE, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(9):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=4.8, color=ink_on(M[i, j], 1))
    for y in np.arange(step, len(rows), step) - 0.5:
        ax.axhline(y, color="white", lw=2)
    ax.set_xticks(range(9), PNAME, rotation=50, ha="right", fontsize=6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6)
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    plab(ax, s)
cax = fig.add_axes([0.30, 0.035, 0.40, 0.012])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("Jaccard overlap of the top-10% hotspots (median over draws)", fontsize=7, color=INK2)
cb.ax.tick_params(labelsize=6)
fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.17)
save(fig, "fig_jaccard_by_province")

# --- factor reliance: pooled permutation + SHAP by algorithm
sh = pd.read_csv(EXP / "02_data" / "runs" / "shap" / "shap_summary.csv")
pp = pr[pr.kind == "perm"].groupby("name")["drop"]
assert (pp.size() == 810).all()
order_p = pp.median().sort_values().index
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"wspace": 0.95})
ax = axs[0]
y = np.arange(len(order_p))
ax.hlines(y, pp.quantile(0.25)[order_p], pp.quantile(0.75)[order_p], color="#2a78d6", lw=1.6)
ax.plot(pp.median()[order_p], y, "o", color="#2a78d6", ms=4, mec="white", mew=0.7)
ax.axvline(0, color=AXIS, lw=0.8)
ax.set_yticks(y, [FEATN[k] for k in order_p], fontsize=6.5); ax.set_xlabel("Loss of held-out AUC when permuted")
ax.grid(axis="y", visible=False); plab(ax, "a")
ax = axs[1]
medv = sh.groupby(["feature", "learner"])["mean_abs_shap"].median().unstack()
sign = sh.groupby("feature")["direction_spearman"].apply(lambda s: "+" if (s > 0).sum() > len(s) / 2 else "−")
order_s = medv.max(axis=1).sort_values().index
y = np.arange(len(order_s))
for k, L in enumerate(("RF", "XGB", "LR")):
    ax.plot(medv.loc[order_s, L], y + (k - 1) * 0.18, "o", color=COL[L], ms=4, mec="white", mew=0.6, label=NAME[L])
ax.set_yticks(y, [f"{FEATN[f]} ({sign[f]})" for f in order_s], fontsize=6.5)
ax.set_xlabel("Mean |SHAP|, probability scale\n(median over six rings)")
ax.grid(axis="y", visible=False); ax.legend(loc="lower right", fontsize=6.5); plab(ax, "b")
fig.subplots_adjust(left=0.24, right=0.98, top=0.93, bottom=0.14)
save(fig, "fig_factor_reliance")

# --- SHAP by ring heatmap, one panel per algorithm
order_h = sh.groupby("feature")["mean_abs_shap"].median().sort_values(ascending=False).index
vmax = sh["mean_abs_shap"].max() * 100
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.9), sharey=True, gridspec_kw={"wspace": 0.06})
for ax, L, s in zip(axs, ("RF", "XGB", "LR"), "abc"):
    d = sh[sh.learner == L].set_index(["feature", "radius"])
    M = np.array([[d.loc[(f, r), "mean_abs_shap"] * 100 for r in RAD] for f in order_h])
    D = np.array([[d.loc[(f, r), "direction_spearman"] for r in RAD] for f in order_h])
    im = ax.imshow(M, cmap=BLUE, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:.1f}{'+' if D[i, j] > 0 else '−'}", ha="center", va="center", fontsize=5,
                    color=ink_on(M[i, j], vmax))
    ax.axvline(0.5, color="white", lw=2)
    ax.set_xticks(range(6), [x + ("" if x == "R∞" else " km") for x in XL], fontsize=6)
    ax.set_yticks(range(len(order_h)), [FEATN[f] for f in order_h], fontsize=6.5)
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    plab(ax, s)
cax = fig.add_axes([0.40, 0.06, 0.40, 0.025])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("Mean |SHAP| × 100, probability scale (sign: direction)", fontsize=7, color=INK2)
cb.ax.tick_params(labelsize=6)
fig.subplots_adjust(left=0.22, right=0.99, top=0.93, bottom=0.22)
save(fig, "fig_shap_by_radius")

# ------------------------------------------------------------------ graphical abstract (Elsevier spec: >= 531 x 1328 px,
# aspect ~2.5, readable at 5 x 13 cm). Every number is read from the locked outputs, none typed by hand.
GA_W, GA_H = 13.28 / 2.54, 5.31 / 2.54                      # inches
plt.rcParams["font.family"] = "Arial"                     # MDPI recommends Arial/Calibri/Times for GA text
fig = plt.figure(figsize=(GA_W, GA_H))
fig.patch.set_facecolor("white")
fs, fsm = 5.6, 5.0


def panel_title(x, txt):
    fig.text(x, 0.965, txt, fontsize=7.5, fontweight="bold", color=INK, ha="center", va="top")


for xline in (0.315, 0.655):                                   # thin separators between panels
    fig.add_artist(Line2D([xline, xline], [0.17, 0.95], color=GRID, lw=0.6))

# panel 1: design (x 0.01-0.30)
panel_title(0.157, "a")
ax = fig.add_axes([0.012, 0.17, 0.29, 0.70]); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0.05, 9.4, "Background ring (km)", fontsize=fsm, color=INK2, va="center")
for i, lab in enumerate(XL):
    ax.add_patch(FancyBboxPatch((0.05 + i * 1.62, 7.75), 1.38, 0.95, boxstyle="round,pad=0.04,rounding_size=0.2",
                                fc="#e8f1fc", ec="#256abf", lw=0.5))
    ax.text(0.74 + i * 1.62, 8.22, lab, fontsize=fsm, ha="center", va="center", color=INK)
ax.text(4.9, 6.85, "×", fontsize=7, ha="center", va="center", color=INK2)
ax.text(0.05, 6.0, "Algorithm", fontsize=fsm, color=INK2, va="center")
for i, L in enumerate(("RF", "XGB", "LR")):
    ax.add_patch(FancyBboxPatch((0.05 + i * 3.25, 4.35), 2.95, 0.95, boxstyle="round,pad=0.04,rounding_size=0.2",
                                fc="white", ec=COL[L], lw=0.9))
    ax.text(1.52 + i * 3.25, 4.82, SHORT[L], fontsize=fsm, ha="center", va="center", color=INK)
for y, line in ((3.0, "Leave-one-province-out,"), (2.25, "shared test sets"),
                (1.25, "Maps: SSIM + top-10% overlap")):
    ax.text(0.05, y, line, fontsize=fsm, color=INK2, va="center")
ax.text(0.05, 0.25, "Pre-specified, hash-locked", fontsize=fsm, color="#256abf", va="center", fontweight="bold")

# panel 2: at 10 km - AUC of the three algorithms vs hotspot overlap of the three pairs (not shown this way in any figure)
panel_title(0.49, "b")
cenA = fold.groupby(["learner", "radius", "seed"])["auc"].median().groupby(["learner", "radius"]).median()
ax = fig.add_axes([0.355, 0.24, 0.285, 0.60])
auc10 = [cenA.loc[(L, "10km")] for L in ("RF", "XGB", "LR")]
j10 = [provvals("across_learners", "10km", f"{a_}_10km", f"{b_}_10km", "jaccard_top10")[0] for a_, b_ in PAIRS]
xa, xb = np.arange(3), np.arange(3) + 4
ax.bar(xa, auc10, color=[COL[L] for L in ("RF", "XGB", "LR")], width=0.75)
ax.bar(xb, j10, color=[PCOL[p] for p in PAIRS], width=0.75)
for x_, v in zip(list(xa) + list(xb), auc10 + j10):
    ax.text(x_, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=4.8, color=INK)
ax.set_xticks(list(xa) + list(xb), ["RF", "XGB", "LR", "RF–\nXGB", "RF–\nLR", "XGB–\nLR"], fontsize=4.6, rotation=0)
ax.set_ylim(0, 1.22); ax.set_yticks([0, 0.5, 1.0]); ax.tick_params(labelsize=fsm, length=2)
ax.grid(axis="x", visible=False)
ax.text(1, -0.36, "Held-out AUC", transform=ax.get_xaxis_transform(), ha="center", fontsize=fsm, color=INK2)
ax.text(5, -0.36, "Hotspot overlap", transform=ax.get_xaxis_transform(), ha="center", fontsize=fsm, color=INK2)
ax.text(6.4, 1.17, "10 km ring", ha="right", va="center", fontsize=fsm, color=INK2)

# panel 3: area, km2 - one map's top 10% vs consensus core vs ever a hotspot (75 maps, rings >= 5 km)
panel_title(0.83, "c")
one_map = 0.1 * valid.sum() * 0.25
core_km2 = float((fB >= 0.9).sum()) * 0.25
dep_km2 = float(((fB > 0) & (fB < 0.9)).sum()) * 0.25
ax = fig.add_axes([0.77, 0.24, 0.215, 0.60])
labels = ["One map's\ntop 10%", "Consensus\ncore", "Hotspot in\nany map"]
vals = [one_map, core_km2, core_km2 + dep_km2]
cols = ["#9ec5f4", "#184f95", "#f0a830"]
ax.barh([2, 1, 0], vals, color=cols, height=0.62)
for y_, v in zip([2, 1, 0], vals):
    ax.text(v + 300, y_, f"{num(v)} km²", va="center", fontsize=fsm, color=INK)
ax.set_yticks([2, 1, 0], labels, fontsize=fsm)
ax.set_xlim(0, max(vals) * 1.62); ax.set_xticks([]); ax.tick_params(length=0, pad=2)
for sp in ("top", "right", "bottom"):
    ax.spines[sp].set_visible(False)
ax.grid(False)
ax.text(-0.40, -0.16, "75 maps, rings ≥ 5 km (exploratory)", transform=ax.transAxes, fontsize=4.8, color=MUTED)
for ext_ in ("png", "pdf"):
    fig.savefig(HERE / f"graphical_abstract.{ext_}", dpi=762)
fig.savefig(HERE / "graphical_abstract.tif", dpi=762, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
im = Image.open(HERE / "graphical_abstract.tif").convert("RGB")          # flatten RGBA -> RGB for artwork checks
im.save(HERE / "graphical_abstract.tif", compression="tiff_lzw", dpi=(762, 762))
print("graphical abstract px:", im.size, f"core {core_km2:.1f} km2, dependent {dep_km2:.1f} km2, one map {one_map:.1f}")
plt.rcParams["font.family"] = "Segoe UI"
print("done")
