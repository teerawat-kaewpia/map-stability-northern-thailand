"""
Map-stability experiment - figures (Fig. 1-5) from the frozen outputs. No model is fitted or refitted;
every value comes from 02_data (hash-recorded in 04_notes/results_manifest_2026-10-01.json).

Output: 03_figures/fig{n}_*.png and .tif, 1000 dpi; TIFF is RGB with LZW compression.
Colour (validated with the dataviz skill's validate_palette.js, light mode):
  learners RF/XGB/LR = categorical slots 1-3 (#2a78d6 / #eb6834 / #1baf7a), adjacent + all-pairs PASS;
  aqua is below 3:1 contrast, so every learner series carries a legend and a direct label;
  difference map RF-only / LR-only / both = #2a78d6 / #1baf7a / #4a3aa7, all-pairs PASS;
  magnitude (suitability, Jaccard) = one-hue blue ramp, fixed 0-1 scale, no per-map rescaling.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
Image.MAX_IMAGE_PIXELS = None
EXP = Path(__file__).resolve().parents[1]
RUNS, MAPS, FIG = EXP / "02_data" / "runs", EXP / "02_data" / "maps", EXP / "03_figures"
FIG.mkdir(parents=True, exist_ok=True)
DPI = 1000

# ---- design tokens (reference palette, light) ----
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
LEARNER_COL = {"RF": "#2a78d6", "XGB": "#eb6834", "LR": "#1baf7a"}
LEARNER_NAME = {"RF": "Random forest", "XGB": "XGBoost", "LR": "Logistic regression"}
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
RADII = ["1km", "5km", "10km", "20km", "25km", "Rinf"]
RLAB = {"1km": "1", "5km": "5", "10km": "10", "20km": "20", "25km": "25", "Rinf": "∞"}
LEARNERS = ["RF", "XGB", "LR"]

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.titlesize": 8, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "legend.fontsize": 7, "text.color": INK})


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5); ax.set_axisbelow(True)


def save(fig, name):
    png, tif = FIG / f"{name}.png", FIG / f"{name}.tif"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    im.save(tif, compression="tiff_lzw", dpi=(DPI, DPI))
    Image.open(png).convert("RGB").save(png, dpi=(DPI, DPI), optimize=True)   # PNG with dpi tag
    w, h = im.size
    print(f"{name}: {w} x {h} px at {DPI} dpi ({w / DPI:.2f} x {h / DPI:.2f} in)")


# ---- map geometry: rebuilt exactly as the runner built it (locked boundary, 500 m) ----
lock = json.loads((EXP / "00_predeclaration" / "input_lock_2026-10-01.json").read_text(encoding="utf-8"))
BND = gpd.read_file(next(i["file"] for i in lock["inputs"] if i["file"].endswith(".gpkg"))).to_crs("EPSG:32647")
BND = BND.sort_values("ADM1_PCODE")
PROV_EN = dict(zip(BND["ADM1_PCODE"], BND["ADM1_EN"]))
VALID = np.load(MAPS / "valid_mask.npy")
H, W = VALID.shape
from shapely.ops import unary_union
minx, miny, maxx, maxy = unary_union(list(BND.geometry)).bounds
minx, miny = np.floor(minx / 500) * 500, np.floor(miny / 500) * 500
maxy = np.ceil(maxy / 500) * 500
EXTENT = (minx, minx + W * 500, maxy - H * 500, maxy)


def grid(v):
    a = np.full(VALID.shape, np.nan, "float32"); a[VALID] = v; return a


def map_axes(ax):
    BND.boundary.plot(ax=ax, color=INK2, linewidth=0.25)
    ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scale_bar(ax, km=100):
    x0, y0 = EXTENT[0] + 0.05 * (EXTENT[1] - EXTENT[0]), EXTENT[2] + 0.04 * (EXTENT[3] - EXTENT[2])
    ax.plot([x0, x0 + km * 1000], [y0, y0], color=INK, linewidth=1.0, solid_capstyle="butt")
    ax.text(x0 + km * 500, y0 + 9000, f"{km} km", ha="center", va="bottom", fontsize=6, color=INK2)


# ================================================================ Fig 1 - discrimination by radius
acr = pd.read_csv(RUNS / "lopo_fold_median_across_seeds.csv")
def spread(vals, gap):
    """Shift end-label positions apart by at least `gap`, keeping their order (simple label repel)."""
    order = np.argsort(vals); out = np.array(vals, float)
    for i in range(1, len(order)):
        if out[order[i]] - out[order[i - 1]] < gap:
            out[order[i]] = out[order[i - 1]] + gap
    return out - (out.mean() - np.mean(vals))


fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.28})
for ax, met, lab in [(axes[0], "auc", "AUC (held-out province)"), (axes[1], "tss", "TSS (held-out province)")]:
    x = np.arange(len(RADII)); ends = []
    for k, L in enumerate(LEARNERS):
        d = acr[acr.learner == L].set_index("radius").loc[RADII]
        xs = x + (k - 1) * 0.14
        ax.vlines(xs, d[f"{met}_median_min"], d[f"{met}_median_max"], color=LEARNER_COL[L], linewidth=1.2)
        ax.plot(xs, d[f"{met}_median_median"], color=LEARNER_COL[L], linewidth=1.2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.6, label=LEARNER_NAME[L], zorder=3)
        ends.append((L, xs[-1], d[f"{met}_median_median"].iloc[-1]))
    lo, hi = ax.get_ylim()
    ys = spread([e[2] for e in ends], 0.06 * (hi - lo))
    for (L, xe, ye), yl in zip(ends, ys):                                # direct labels, repelled
        ax.plot([xe + 0.06, xe + 0.30], [ye, yl], color=AXIS, linewidth=0.5)
        ax.text(xe + 0.34, yl, L, color=INK2, fontsize=6.5, va="center")
    ax.set_xticks(x, [RLAB[r] for r in RADII]); ax.set_xlim(-0.4, len(RADII) - 0.05)
    ax.set_xlabel("Outer radius of the training-background ring (km)"); ax.set_ylabel(lab)
    style(ax)
axes[0].set_title("a  Discrimination", loc="left"); axes[1].set_title("b  True skill statistic", loc="left")
axes[0].legend(loc="lower right", handlelength=1.6)
fig.text(0.01, -0.05, "Points: median over 5 seeds of the 9-fold median; bars: range of the 5 seed-level fold medians "
         "(not a confidence interval). Cohen's κ equals TSS here (1:1 test sets).", fontsize=6, color=MUTED)
save(fig, "fig1_discrimination_by_radius")

# ================================================================ Fig 2 - suitability maps, three learners at 10 km
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.03})
fig.subplots_adjust(top=0.93, bottom=0.20, left=0.01, right=0.99)
for ax, L, tag in zip(axes, LEARNERS, "abc"):
    ax.imshow(grid(np.load(MAPS / f"{L}_10km_s42.npy")), cmap=SEQ, vmin=0, vmax=1, extent=EXTENT,
              interpolation="nearest", origin="upper")
    map_axes(ax); ax.set_title(f"{tag}  {LEARNER_NAME[L]}", loc="left")
scale_bar(axes[0])
cax = fig.add_axes([0.30, 0.12, 0.40, 0.03])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal")
cb.set_label("Predicted suitability score (0-1, same scale in every panel)", color=INK2, fontsize=6.5, labelpad=2)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=2)
fig.text(0.01, -0.03, "10 km ring, seed 42, models fitted on all nine provinces; 500 m grid. Scores are relative suitability "
         "(1:1 presence-background), not probabilities of a firm being present.", fontsize=6, color=MUTED)
save(fig, "fig2_suitability_maps_10km")

# ================================================================ Fig 3 - where the top 10% moves: RF vs LR at 10 km
cls = np.load(MAPS / "diff_top10_RF_10km_vs_LR_10km_s42.npy")          # 2 both, 1 RF only, -1 LR only, 0 neither
codes = {"neither": 0, "LR only": -1, "RF only": 1, "both": 2}
n = {k: int((cls == v).sum()) for k, v in codes.items()}
jac = n["both"] / (n["both"] + n["RF only"] + n["LR only"])
cmap = ListedColormap([LEARNER_COL["LR"], NEUTRAL, LEARNER_COL["RF"], "#4a3aa7"])
norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
fig, ax = plt.subplots(figsize=(4.6, 5.4))
ax.imshow(grid(cls.astype("float32")), cmap=cmap, norm=norm, extent=EXTENT, interpolation="nearest", origin="upper")
map_axes(ax); scale_bar(ax)
ax.set_title(f"Top-10% cells, random forest vs logistic regression (10 km, seed 42)\nstudy-wide Jaccard = {jac:.2f}",
             loc="left", fontsize=7.5)
km2 = lambda c: f"{c * 0.25:,.0f} km²"
ax.legend(handles=[Patch(color="#4a3aa7", label=f"Both models  ({km2(n['both'])})"),
                   Patch(color=LEARNER_COL["RF"], label=f"Random forest only  ({km2(n['RF only'])})"),
                   Patch(color=LEARNER_COL["LR"], label=f"Logistic regression only  ({km2(n['LR only'])})"),
                   Patch(facecolor=NEUTRAL, edgecolor=AXIS, linewidth=0.4, label="Neither")],
          loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=2, fontsize=6.5, handlelength=1.2,
          columnspacing=1.2)
ax.text(0.0, -0.14, "Each map's top 10% is a study-wide decile with the same number of cells, so the two "
        "'only' areas are equal by construction.", transform=ax.transAxes, fontsize=5.5, color=MUTED, va="top")
save(fig, "fig3_top10_difference_RF_vs_LR_10km")

# ================================================================ Fig 4 - Jaccard by province (median over 5 seeds)
sm = pd.read_csv(RUNS / "map_similarity_summary_across_seeds.csv")
jac_df = sm[sm.metric == "jaccard_top10"]
provs = list(BND["ADM1_PCODE"])
rows_a = [(f"{L}: {RLAB[r]} km", f"{L}_{r}", f"{L}_10km") for L in LEARNERS for r in RADII if r != "10km"]
rows_b = [(f"{RLAB[r]} km: {a} vs {b}", f"{a}_{r}", f"{b}_{r}") for r in RADII
          for a, b in [("RF", "XGB"), ("RF", "LR"), ("XGB", "LR")]]


def _lum(rgb):
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb[:3]]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def text_on(rgba):
    """Ink or white, whichever has the higher WCAG contrast against the cell colour."""
    lc = _lum(rgba); ink = _lum(matplotlib.colors.to_rgb(INK))
    return "#ffffff" if (1.05) / (lc + 0.05) >= (lc + 0.05) / (ink + 0.05) else INK


def mat(rows):
    return np.array([[jac_df[(jac_df.a == a) & (jac_df.b == b) & (jac_df.province == p)]["median"].iloc[0]
                      for p in provs] for _, a, b in rows])


fig, axes = plt.subplots(1, 2, figsize=(7.2, 6.4), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1]})
for ax, rows, title in [(axes[0], rows_a, "a  Across radii (vs 10 km), same learner"),
                        (axes[1], rows_b, "b  Across learners, same radius")]:
    M = mat(rows)
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5,
                    color=text_on(SEQ(M[i, j])))
    ax.set_xticks(range(len(provs)), [PROV_EN[p] for p in provs], rotation=50, ha="right", fontsize=6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(title, loc="left", fontsize=7.5)
    step = 3 if ax is axes[1] else 5                     # b: 3 pairs per radius; a: 5 radii per learner
    for i in range(step, len(rows), step):
        ax.axhline(i - 0.5, color=SURFACE, linewidth=1.5)
cax = fig.add_axes([0.30, 0.0, 0.40, 0.014])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal")
cb.set_label("Jaccard overlap of the top-10% cells (median over 5 seeds)", color=INK2, fontsize=6.5)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
save(fig, "fig4_jaccard_by_province")

# ================================================================ Fig 5 - factor use
perm = {}
for f in sorted((RUNS / "lopo").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    for L, v in d.get("learners", {}).items():
        for k, (mean, _sd) in v["perm_auc_drop"].items():
            perm.setdefault(k, []).append(mean)
pd_ = pd.DataFrame({k: pd.Series(v) for k, v in perm.items()})
order = pd_.median().sort_values().index.tolist()
shp = pd.read_csv(RUNS / "shap" / "shap_summary.csv")
sh_med = shp.groupby(["learner", "feature"])["mean_abs_shap"].median().unstack(0)
sign = shp.groupby("feature")["direction_spearman"].apply(lambda s: "+" if (s > 0).sum() > (s < 0).sum() else "−")
feat_order = sh_med.max(axis=1).sort_values().index.tolist()
PRETTY = {"slope_deg": "Slope", "builtup": "Built-up", "nightlight": "Night-time light",
          "dist_road_km": "Distance to road", "dist_rail_km": "Distance to existing railway",
          "dist_railway_new_km": "Distance to planned DCR-CK line", "dist_water_km": "Distance to water",
          "dist_cbd_km": "Distance to provincial centre", "dist_dryport_km": "Distance to dry port",
          "dist_border_CK_km": "Distance to Chiang Khong border", "forest": "Forest-area layer",
          "builtup+nightlight": "Built-up + night-time light (joint)"}
pretty = lambda k: PRETTY.get(k, k)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"wspace": 0.95})
ax = axes[0]; y = np.arange(len(order))
q1, med, q3 = pd_[order].quantile(0.25), pd_[order].median(), pd_[order].quantile(0.75)
ax.hlines(y, q1, q3, color=BLUE_RAMP[7], linewidth=1.4)
ax.plot(med, y, "o", color=BLUE_RAMP[7], markersize=4, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3)
ax.axvline(0, color=AXIS, linewidth=0.6)
ax.set_yticks(y, [pretty(k) for k in order], fontsize=6.5); ax.set_xlabel("Drop in held-out AUC when permuted")
ax.set_title("a  Permutation importance", loc="left"); style(ax, grid_axis="x")
ax.text(1.0, -0.22, "Median and IQR over 810 fold × radius × seed × learner units", transform=ax.transAxes,
        ha="right", fontsize=5.5, color=MUTED)
ax = axes[1]; y = np.arange(len(feat_order))
for k, L in enumerate(LEARNERS):
    ax.plot(sh_med.loc[feat_order, L], y + (k - 1) * 0.22, "o", color=LEARNER_COL[L], markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=0.6, label=LEARNER_NAME[L], zorder=3)
ax.set_yticks(y, [f"{pretty(f)}  ({sign[f]})" for f in feat_order], fontsize=6.5)
ax.set_xlabel("Mean |SHAP|, probability scale\n(median over 6 radii)")
ax.set_title("b  SHAP contribution by learner", loc="left"); style(ax, grid_axis="x")
ax.legend(loc="lower right", fontsize=6.5, handletextpad=0.2)
ax.text(1.0, -0.26, "(+/−) = sign of the value-SHAP correlation in most of the 18 models; descriptive, not causal",
        transform=ax.transAxes, ha="right", fontsize=5.5, color=MUTED)
save(fig, "fig5_factor_use")
print("done")
