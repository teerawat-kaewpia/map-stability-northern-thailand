"""
Map-stability experiment - Fig. 9: AUC, TSS and Cohen's kappa side by side, per learner x radius, from the
frozen leave-one-province-out results (02_data/runs/lopo_metrics_by_fold.csv, hash-recorded in
04_notes/results_manifest_2026-10-01.json). Nothing is refitted. Descriptive: no test is run.
Each learner x radius has 45 values (9 held-out provinces x 5 seeds): point = median, thick bar = IQR,
thin bar = min-max. Kappa equals TSS to floating-point precision because every test set holds equal numbers
of presence and background points; the script asserts this and the figure says so.
Colours: learners = categorical slots 1-3 (#2a78d6 / #eb6834 / #1baf7a), as in Fig. 1-5.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
EXP = Path(__file__).resolve().parents[1]
RUNS, FIG, DPI = EXP / "02_data" / "runs", EXP / "03_figures", 1000

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LEARNER_COL = {"RF": "#2a78d6", "XGB": "#eb6834", "LR": "#1baf7a"}
LEARNER_NAME = {"RF": "Random forest", "XGB": "XGBoost", "LR": "Logistic regression"}
LEARNERS = ["RF", "XGB", "LR"]
RADII = ["1km", "5km", "10km", "20km", "25km", "Rinf"]
RLAB = {"1km": "1", "5km": "5", "10km": "10", "20km": "20", "25km": "25", "Rinf": "∞"}
plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.titlesize": 8, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "text.color": INK})


def save(fig, name):
    png, tif = FIG / f"{name}.png", FIG / f"{name}.tif"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.04); plt.close(fig)
    im = Image.open(png).convert("RGB")
    im.save(tif, compression="tiff_lzw", dpi=(DPI, DPI)); im.save(png, dpi=(DPI, DPI), optimize=True)
    print(f"{name}: {im.size[0]} x {im.size[1]} px at {DPI} dpi")


d = pd.read_csv(RUNS / "lopo_metrics_by_fold.csv")
assert (d.groupby(["learner", "radius"]).size() == 45).all()
gap = (d["kappa"] - d["tss"]).abs().max()
assert gap < 1e-12, gap
print(f"max |kappa - TSS| = {gap:.1e}")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.30})
x = np.arange(len(RADII))
for ax, (met, title, ylab) in zip(axes, [("auc", "a  AUC", "AUC (held-out province)"),
                                          ("tss", "b  TSS", "TSS (held-out province)"),
                                          ("kappa", "c  Cohen's κ", "κ (held-out province)")]):
    for k, L in enumerate(LEARNERS):
        g = d[d.learner == L].groupby("radius")[met]
        q = pd.DataFrame({"lo": g.min(), "q1": g.quantile(0.25), "md": g.median(), "q3": g.quantile(0.75),
                          "hi": g.max()}).loc[RADII]
        xs = x + (k - 1) * 0.22
        ax.vlines(xs, q.lo, q.hi, color=LEARNER_COL[L], linewidth=0.6, alpha=0.8)
        ax.vlines(xs, q.q1, q.q3, color=LEARNER_COL[L], linewidth=2.6)
        ax.plot(xs, q.md, linestyle="none", marker="o", markersize=3.6, color=LEARNER_COL[L],
                markeredgecolor=SURFACE, markeredgewidth=0.6, label=LEARNER_NAME[L], zorder=3)
    ax.axvspan(-0.5, 0.5, color="#f0efec", zorder=0)                    # set the 1 km ring apart
    ax.set_xticks(x, [RLAB[r] for r in RADII]); ax.set_xlim(-0.5, len(RADII) - 0.5)
    ax.set_xlabel("Outer radius of the training-\nbackground ring (km)"); ax.set_ylabel(ylab)
    ax.set_title(title, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
axes[2].set_ylim(axes[1].get_ylim())                                     # same scale as TSS: identical values
axes[2].text(0.97, 0.04, "identical to TSS\n(1:1 test sets)", transform=axes[2].transAxes, ha="right",
             va="bottom", fontsize=6, color=INK2)
axes[0].legend(loc="lower right", fontsize=6.5, handlelength=1.0)
fig.text(0.01, -0.10, "Each learner x radius: 45 values (9 held-out provinces x 5 seeds). Point = median; thick bar = "
         "interquartile range; thin bar = minimum-maximum. TSS and κ at the frozen max-TSS threshold.\n"
         "κ equals TSS to floating-point precision (max difference 2e-16) because each test set has equal "
         "presence and background counts. Shaded: 1 km ring. Descriptive; no significance test.",
         fontsize=5.8, color=MUTED, va="top")
save(fig, "fig9_auc_tss_kappa")
print("done")
