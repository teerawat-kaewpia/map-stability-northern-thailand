"""
Map-stability experiment - Fig. 8a-c, POST-HOC descriptive: SHAP dependence shape for ALL 11 covariates,
one figure per learner, every radius overlaid. Same locked seed-42 SHAP outputs as Fig. 6-7 (02_data/runs/shap);
nothing is refitted. Styling, palette and save routine as make_figures_shap.py (imported read-only constants
are re-declared here so that script's outputs are not re-rendered).
Continuous covariates: mean SHAP per decile (rank-split). Binary covariates (forest, built-up): mean SHAP at
0 and 1, with the number of explained cells in each group printed under the axis.
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
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
Image.MAX_IMAGE_PIXELS = None
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "01_scripts"))
import run_map_stability as R                                 # read-only: grid rebuild for covariate values
SH, FIG, DPI = EXP / "02_data" / "runs" / "shap", EXP / "03_figures", 1000

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RADII = ["1km", "5km", "10km", "20km", "25km", "Rinf"]
RLAB = {"1km": "1 km", "5km": "5 km", "10km": "10 km", "20km": "20 km", "25km": "25 km", "Rinf": "∞"}
RCOL = {"5km": "#2a78d6", "10km": "#eda100", "20km": "#1baf7a", "25km": "#4a3aa7", "Rinf": "#e34948"}
RMRK = {"1km": "o", "5km": "s", "10km": "^", "20km": "D", "25km": "v", "Rinf": "P"}
LEARNERS = {"RF": ("a", "Random forest"), "XGB": ("b", "XGBoost"), "LR": ("c", "Logistic regression")}
PRETTY = {"slope_deg": "Slope", "builtup": "Built-up", "nightlight": "Night-time light",
          "dist_road_km": "Distance to road", "dist_rail_km": "Distance to existing railway",
          "dist_railway_new_km": "Distance to planned DCR-CK line", "dist_water_km": "Distance to water",
          "dist_cbd_km": "Distance to provincial centre", "dist_dryport_km": "Distance to dry port",
          "dist_border_CK_km": "Distance to Chiang Khong border", "forest": "Forest-area layer"}
plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.titlesize": 7, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "text.color": INK})


def save(fig, name):
    png, tif = FIG / f"{name}.png", FIG / f"{name}.tif"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.04); plt.close(fig)
    im = Image.open(png).convert("RGB")
    im.save(tif, compression="tiff_lzw", dpi=(DPI, DPI)); im.save(png, dpi=(DPI, DPI), optimize=True)
    print(f"{name}: {im.size[0]} x {im.size[1]} px at {DPI} dpi")


summ = pd.read_csv(SH / "shap_summary.csv")
feat_order = summ.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).index.tolist()
G = R.build_grid(R.Data(json.loads(R.LOCK.read_text(encoding="utf-8"))))
X = G["X"].iloc[np.load(SH / "explained_cells.npy")].reset_index(drop=True)

for L, (tag, lname) in LEARNERS.items():
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 6.0), gridspec_kw={"hspace": 0.62, "wspace": 0.42})
    for k, f in enumerate(feat_order):
        ax, fj = axes.flat[k], R.FEATURES.index(f)
        binary = X[f].nunique() <= 2
        grp = X[f].astype(int) if binary else pd.qcut(X[f].rank(method="first"), 10, labels=False) + 1
        for r in RADII:
            y = pd.Series(np.load(SH / f"shap_{L}_{r}.npy")[:, fj]).groupby(grp.values).mean()
            kw = dict(color=INK2, linestyle=(0, (3, 2)), linewidth=1.0) if r == "1km" else dict(color=RCOL[r], linewidth=1.2)
            ax.plot(y.index, y.values, marker=RMRK[r], markersize=3.5 if binary else 2.6, label=RLAB[r], **kw)
        ax.axhline(0, color=AXIS, linewidth=0.6)
        if binary:
            n = grp.value_counts()
            ax.set_xticks([0, 1], [f"0\n(n={n.get(0, 0):,})", f"1\n(n={n.get(1, 0):,})"], fontsize=5.5)
            ax.set_xlim(-0.4, 1.4)
        else:
            ax.set_xticks([1, 5.5, 10], ["low", "decile", "high"], fontsize=5.5)
        ax.tick_params(labelsize=5.5, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.4); ax.set_axisbelow(True)
        ax.set_title(PRETTY[f], loc="left", fontsize=6.5)
        if k % 4 == 0:
            ax.set_ylabel("mean SHAP", fontsize=6.5)
    lax = axes.flat[-1]; lax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    lax.legend(h, l, loc="center left", fontsize=6.5, handlelength=2.4, title="Outer radius of the\ntraining-background ring",
               title_fontsize=6.5, alignment="left")
    fig.suptitle(f"{tag}  {lname}: SHAP dependence for all 11 covariates", x=0.09, ha="left", y=0.955,
                 fontsize=8, fontweight="semibold", color=INK)
    fig.text(0.09, 0.025, "Mean SHAP (probability scale) per decile of the covariate across the 1,000 explained cells; binary layers at 0/1 "
             "with group sizes. Panels ordered by mean |SHAP|;\neach panel has its own y-axis. Night-time light is 0 in 77.6% of cells, "
             "so its lower deciles are tied values split by rank only. Built-up = 1 in 9 cells only. Seed 42; post-hoc, descriptive."
             + ("\nLogistic-regression SHAP is a sampled estimate (PermutationExplainer)." if L == "LR" else ""),
             fontsize=5.5, color=MUTED, va="top")
    save(fig, f"fig8{tag}_shap_dependence_all_{L}")
print("done")
