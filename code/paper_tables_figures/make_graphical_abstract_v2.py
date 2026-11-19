"""Graphical abstract, plain-language redesign after Reviewer 4 (approved 2026-11-05). Panel a is the design panel
of the earlier graphical abstract.

Three findings left to right, one take-home line underneath:
  a  accuracy is nearly the same: held-out AUC of the three algorithms at the 10 km ring, province-wide tests
  b  but the top-ranked areas differ: how many of the three algorithms place each cell in their top 10% (Figure 5d)
  c  tested near factories, the ranking of rings changes (post hoc): AUC of the 1 km and 10 km rings on the
     province-wide test and on test points 0.5-1 km from factories (Table S21)
Every number is computed here from the locked outputs, none typed by hand. Writes graphical_abstract.png/.pdf/.tif
(called at the end of make_paper_figures.py, so a full figure rebuild keeps this version).
Usage: py -3.11 make_graphical_abstract_v2.py"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
PD = HERE.parent
RFRP = PD.parent
EXP = RFRP / "exp_map_stability"
BND = RFRP / "paper_B" / "03_results" / "rasters" / "boundary_9prov_32647.gpkg"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COL = {"RF": "#2a78d6", "XGB": "#eb6834", "LR": "#1baf7a"}
NAME = {"RF": "Random forest", "XGB": "XGBoost", "LR": "Logistic regression"}
AGREE = ListedColormap(["#ebeae4", "#f6c27a", "#d9731f", "#184f95"])          # as Figure 5d
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "text.color": INK, "axes.edgecolor": "#c3c2b7",
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------------------------------------- numbers
f = pd.read_csv(EXP / "02_data/runs/lopo_metrics_by_fold.csv", encoding="utf-8-sig")
auc = f.groupby(["learner", "radius", "seed"]).auc.median().groupby(["learner", "radius"]).median()
u = pd.read_csv(PD / "08_revision_posthoc/outputs/P1_auc_by_unit.csv")
near = u.groupby(["learner", "radius", "seed"]).auc_near1km.median().groupby(["learner", "radius"]).median()
maps = EXP / "02_data" / "maps"
m10 = {L: np.load(maps / f"{L}_10km_s42.npy") for L in ("RF", "XGB", "LR")}
agree = sum((v >= np.quantile(v, 0.9)).astype(int) for v in m10.values())
one_map = len(agree) * 0.1 * 0.25
all3 = float((agree == 3).sum()) * 0.25
valid = np.load(maps / "valid_mask.npy")
H, W = valid.shape
bnd = gpd.read_file(BND).to_crs(32647)
minx, miny, maxx, maxy = bnd.union_all().bounds
minx, maxy = np.floor(minx / 500) * 500, np.ceil(maxy / 500) * 500
ext = (minx, minx + W * 500, maxy - H * 500, maxy)
g = np.full((H, W), np.nan, np.float32); g[valid] = agree

# ---------------------------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(16 / 2.54, 8 / 2.54))
fig.patch.set_facecolor("white")


def head(x, lab, txt, sub):
    fig.text(x, 0.955, lab, fontsize=9, fontweight="bold", ha="center", va="top")
    fig.text(x, 0.885, txt, fontsize=8, fontweight="bold", ha="center", va="top", color=INK)
    fig.text(x, 0.815, sub, fontsize=6, ha="center", va="top", color=INK2)


# a: design, as panel a of the earlier graphical abstract (author's choice 2026-11-05)
head(0.165, "a", "What was compared", "Pre-specified experiment, nine provinces")
ax = fig.add_axes([0.03, 0.20, 0.27, 0.56]); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
fsm = 6.3
ax.text(0.05, 9.4, "Background ring (km)", fontsize=fsm, color=INK2, va="center")
for i, lab in enumerate(["1", "5", "10", "20", "25", "R∞"]):
    ax.add_patch(FancyBboxPatch((0.05 + i * 1.62, 7.75), 1.38, 0.95, boxstyle="round,pad=0.04,rounding_size=0.2",
                                fc="#e8f1fc", ec="#256abf", lw=0.5))
    ax.text(0.74 + i * 1.62, 8.22, lab, fontsize=fsm, ha="center", va="center", color=INK)
ax.text(4.9, 6.85, "×", fontsize=8, ha="center", va="center", color=INK2)
ax.text(0.05, 6.0, "Algorithm", fontsize=fsm, color=INK2, va="center")
for i, L in enumerate(("RF", "XGB", "LR")):
    ax.add_patch(FancyBboxPatch((0.05 + i * 3.25, 4.35), 2.95, 0.95, boxstyle="round,pad=0.04,rounding_size=0.2",
                                fc="white", ec=COL[L], lw=0.9))
    ax.text(1.52 + i * 3.25, 4.82, {"RF": "RF", "XGB": "XGBoost", "LR": "LR"}[L], fontsize=fsm, ha="center",
            va="center", color=INK)
for y, line in ((3.0, "Leave-one-province-out,"), (2.25, "shared test sets"),
                (1.25, "Maps: SSIM + top-10% overlap")):
    ax.text(0.05, y, line, fontsize=fsm, color=INK2, va="center")
ax.text(0.05, 0.25, "Pre-specified, hash-locked", fontsize=fsm, color="#256abf", va="center", fontweight="bold")
Ls = ["RF", "XGB", "LR"]
v = [auc[(L, "10km")] for L in Ls]

# b: top-ranked areas differ
head(0.50, "b", "Similar accuracy, different top areas",
     f"10 km ring: AUC {min(v):.3f}–{max(v):.3f}, yet the top 10% differ")
ax = fig.add_axes([0.36, 0.20, 0.28, 0.58])
ax.imshow(g, cmap=AGREE, vmin=0, vmax=3, extent=ext, interpolation="nearest")
bnd.boundary.plot(ax=ax, color=INK2, linewidth=0.25)
ax.set_axis_off(); ax.set_aspect("equal")
fig.text(0.50, 0.215, f"Each map's top 10%: {one_map:.0f} km². All three agree on {all3:.0f} km².", ha="center",
         fontsize=6, color=INK2)
fig.legend([Patch(color=AGREE(3)), Patch(color=AGREE(2)), Patch(color=AGREE(1))],
           ["All three algorithms", "Two algorithms", "One algorithm"], loc="lower center", ncol=3, fontsize=5.6,
           bbox_to_anchor=(0.50, 0.115), frameon=False, handlelength=1.0, columnspacing=0.8)

# c: ranking changes near factories (post hoc)
head(0.835, "c", "Near factories, the ranking changes", "AUC of the 1 km and 10 km rings (post hoc test)")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=COL[L], marker="o", ms=3, lw=1.2) for L in Ls], ["RF", "XGBoost", "LR"],
           loc="lower center", ncol=3, fontsize=5.8, bbox_to_anchor=(0.835, 0.17), frameon=False,
           handlelength=1.4, columnspacing=0.8)
for k, (title, src) in enumerate((("Province-wide test", auc), ("Test 0.5–1 km\nfrom factories", near))):
    ax = fig.add_axes([0.715 + k * 0.145, 0.30, 0.11, 0.42])
    for L in Ls:
        ax.plot([0, 1], [src[(L, "1km")], src[(L, "10km")]], "-o", color=COL[L], lw=1.2, ms=3)
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(0.75, 1.0)
    ax.set_xticks([0, 1], ["1 km", "10 km"], fontsize=6); ax.tick_params(axis="y", labelsize=5.5)
    ax.set_title(title, fontsize=6, color=INK2, pad=3)
    if k:
        ax.set_yticklabels([])
    ax.grid(axis="y", color=GRID, lw=0.5)

# take-home line
fig.add_artist(FancyBboxPatch((0.03, 0.015), 0.94, 0.075, boxstyle="round,pad=0.004,rounding_size=0.01",
                              fc="#eef3f8", ec="none", transform=fig.transFigure))
fig.text(0.5, 0.053, "The maps rank places that resemble existing factory sites, not proven sites for new factories. "
         "Report map agreement beside accuracy.", ha="center", va="center", fontsize=6.6, color=INK)
# approved by the author 2026-11-05: written as the submission graphical abstract (PNG, PDF and TIFF, RGB, LZW)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"graphical_abstract.{ext}", dpi=600)
fig.savefig(HERE / "graphical_abstract.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
from PIL import Image
Image.open(HERE / "graphical_abstract.tif").convert("RGB").save(HERE / "graphical_abstract.tif", compression="tiff_lzw",
                                                                 dpi=(600, 600))
out = HERE / "graphical_abstract.png"
print("wrote", out.name, "| AUC 10 km", {L: round(auc[(L, '10km')], 3) for L in Ls},
      "| near 0.5-1 km 1 km/10 km", {L: (round(near[(L, '1km')], 3), round(near[(L, '10km')], 3)) for L in Ls},
      f"| one map {one_map:.0f} km2, all three {all3:.0f} km2")
