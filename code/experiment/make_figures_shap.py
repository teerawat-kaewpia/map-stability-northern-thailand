"""
Map-stability experiment - SHAP figures (Fig. 6-7), POST-HOC descriptive views of the locked seed-42 SHAP
outputs (02_data/runs/shap, hash-recorded in 04_notes/results_manifest_2026-10-01.json). Nothing is refitted.

Fig 6: mean |SHAP| per covariate x radius, one panel per learner, one shared colour scale (all values are on
       the probability scale, so magnitudes are comparable across learners); each cell carries value and sign.
Fig 7: dependence shape - mean SHAP by decile of the covariate, for four covariates, per learner and radius.
Colour (dataviz validate_palette.js, light): radii 5 km..inf = ordinal blue #86b6ef,#5598e7,#2a78d6,#1c5cab,
#104281 (--ordinal PASS; a 6-step blue ramp fails the step-gap/light-end checks), so the 1 km ring is drawn as a
dashed neutral line - it is also the configuration that behaves differently. Output 1000 dpi PNG + LZW TIFF.
Radius colours (fig7/fig8, revised 2026-10-02 on request - the ordinal blue ramp was hard to tell apart):
categorical #2a78d6,#eda100,#1baf7a,#4a3aa7,#e34948 for 5/10/20/25 km/inf (validate_palette.js --pairs all:
normal-vision PASS 16.3, CVD worst 6.9 = floor band, so every radius also has its own marker shape);
1 km stays a dashed neutral line.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
Image.MAX_IMAGE_PIXELS = None
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "01_scripts"))
import run_map_stability as R                                 # read-only: grid rebuild for covariate values
SH, FIG, DPI = EXP / "02_data" / "runs" / "shap", EXP / "03_figures", 1000

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
      "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
RADII = ["1km", "5km", "10km", "20km", "25km", "Rinf"]
RLAB = {"1km": "1 km", "5km": "5 km", "10km": "10 km", "20km": "20 km", "25km": "25 km", "Rinf": "∞"}
RCOL = {"5km": "#2a78d6", "10km": "#eda100", "20km": "#1baf7a", "25km": "#4a3aa7", "Rinf": "#e34948"}
RMRK = {"1km": "o", "5km": "s", "10km": "^", "20km": "D", "25km": "v", "Rinf": "P"}
LEARNERS = ["RF", "XGB", "LR"]
LNAME = {"RF": "Random forest", "XGB": "XGBoost", "LR": "Logistic regression"}
PRETTY = {"slope_deg": "Slope", "builtup": "Built-up", "nightlight": "Night-time light",
          "dist_road_km": "Distance to road", "dist_rail_km": "Distance to existing railway",
          "dist_railway_new_km": "Distance to planned DCR-CK line", "dist_water_km": "Distance to water",
          "dist_cbd_km": "Distance to provincial centre", "dist_dryport_km": "Distance to dry port",
          "dist_border_CK_km": "Distance to Chiang Khong border", "forest": "Forest-area layer"}
plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.titlesize": 7.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "text.color": INK})


def save(fig, name):
    png, tif = FIG / f"{name}.png", FIG / f"{name}.tif"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.04); plt.close(fig)
    im = Image.open(png).convert("RGB")
    im.save(tif, compression="tiff_lzw", dpi=(DPI, DPI)); im.save(png, dpi=(DPI, DPI), optimize=True)
    print(f"{name}: {im.size[0]} x {im.size[1]} px at {DPI} dpi")


def _lum(rgb):
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb[:3]]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def text_on(rgba):
    lc, ink = _lum(rgba), _lum(matplotlib.colors.to_rgb(INK))
    return "#ffffff" if 1.05 / (lc + 0.05) >= (lc + 0.05) / (ink + 0.05) else INK


summ = pd.read_csv(SH / "shap_summary.csv")
feat_order = summ.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).index.tolist()

# ================================================================ Fig 6 - mean |SHAP| heatmaps
vmax = summ["mean_abs_shap"].max()
fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.0), gridspec_kw={"wspace": 0.08})
for k, (ax, L) in enumerate(zip(axes, LEARNERS)):
    d = summ[summ.learner == L]
    M = d.pivot(index="feature", columns="radius", values="mean_abs_shap").loc[feat_order, RADII]
    S = d.pivot(index="feature", columns="radius", values="direction_spearman").loc[feat_order, RADII]
    ax.imshow(M.values, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, f"{v * 100:.1f}{'+' if S.values[i, j] > 0 else '−'}", ha="center", va="center",
                    fontsize=5, color=text_on(SEQ(v / vmax)))
    ax.set_xticks(range(len(RADII)), [RLAB[r] for r in RADII], fontsize=6)
    ax.set_yticks(range(len(feat_order)), [PRETTY[f] for f in feat_order] if k == 0 else [], fontsize=6)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(f"{'abc'[k]}  {LNAME[L]}", loc="left")
    ax.axvline(0.5, color=SURFACE, linewidth=1.5)                 # set the 1 km ring apart
cax = fig.add_axes([0.36, -0.03, 0.40, 0.022])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, vmax * 100)), cax=cax, orientation="horizontal")
cb.set_label("Mean |SHAP| × 100 (probability-scale points; one scale for all panels)", color=INK2, fontsize=6.5)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=2)
fig.text(0.0, -0.13, "Seed-42 map models, the same 1,000 explained cells. Cell text: value and the sign of the value-SHAP "
         "correlation (+ higher value raises the score). Logistic-regression SHAP is a sampled estimate. Post-hoc, descriptive.",
         fontsize=5.5, color=MUTED)
save(fig, "fig6_shap_by_radius_heatmap")

# ================================================================ Fig 7 - dependence shape by radius
G = R.build_grid(R.Data(json.loads(R.LOCK.read_text(encoding="utf-8"))))
cells = np.load(SH / "explained_cells.npy")
X = G["X"].iloc[cells].reset_index(drop=True)
SHOW = ["slope_deg", "nightlight", "dist_road_km", "forest"]
fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.6), gridspec_kw={"hspace": 0.55, "wspace": 0.42})
for i, L in enumerate(LEARNERS):
    for j, f in enumerate(SHOW):
        ax, fj = axes[i, j], R.FEATURES.index(f)
        binary = X[f].nunique() <= 2
        grp = X[f].astype(int) if binary else pd.qcut(X[f].rank(method="first"), 10, labels=False) + 1
        for r in RADII:
            y = pd.Series(np.load(SH / f"shap_{L}_{r}.npy")[:, fj]).groupby(grp.values).mean()
            kw = dict(color=INK2, linestyle=(0, (3, 2)), linewidth=1.0) if r == "1km" else dict(color=RCOL[r], linewidth=1.2)
            ax.plot(y.index, y.values, marker=RMRK[r], markersize=2.6 if not binary else 3.5, label=RLAB[r], **kw)
        ax.axhline(0, color=AXIS, linewidth=0.6)
        ax.set_xticks([0, 1] if binary else [1, 5.5, 10], ["0", "1"] if binary else ["low", "decile", "high"], fontsize=5.5)
        if binary:
            ax.set_xlim(-0.4, 1.4)
        ax.tick_params(labelsize=5.5, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.4); ax.set_axisbelow(True)
        if i == 0:
            ax.set_title(PRETTY[f], loc="left", fontsize=7)
        if j == 0:
            ax.set_ylabel(f"{LNAME[L]}\nmean SHAP", fontsize=6.5)
axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.62), ncol=6, fontsize=6.5, handlelength=2.2,
                  columnspacing=1.0, title="Outer radius of the training-background ring", title_fontsize=6.5)
fig.text(0.0, 0.015, "Mean SHAP (probability scale) per decile of the covariate across the 1,000 explained cells (forest: 0/1). "
         "Each panel has its own y-axis.\nNight-time light is 0 in most cells, "
         "so its lower deciles are tied values split by rank only. Seed 42; post-hoc, descriptive.", fontsize=5.5, color=MUTED)
save(fig, "fig7_shap_dependence_by_radius")
print("done")
