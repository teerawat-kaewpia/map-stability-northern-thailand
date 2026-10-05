"""Fig 1 (approved by the user 2026-10-05): Thailand locator map, then the study area enlarged, in one image.
Writes fig_study_area.png/.pdf; called from make_paper_figures.py.
Right panel = the Fig 1 code of make_paper_figures.py unchanged. Left panel: OCHA/RTSD ADM1 (same source
as the study boundary, already cited) + Natural Earth 1:10m admin-0 for the neighbouring countries (public domain)."""
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyproj
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle

HERE = Path(__file__).resolve().parent
RFRP = HERE.parents[1]
GIS = RFRP.parents[1]
BND = RFRP / "paper_B" / "03_results" / "rasters" / "boundary_9prov_32647.gpkg"
PRES = RFRP / "data" / "processed" / "analysis_dataset_m3_base_clean.csv"
THA1 = GIS / "Map Data" / "Thai64" / "tha_admbnda_adm1_rtsd_20220121.shp"
NE = GIS / "Energy" / "DER" / "Data" / "01_Admin_Boundaries" / "ne_10m_admin0_countries.zip"

SURFACE, INK, INK2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
LAND, STUDY, SEA = "#f2f1ec", "#a9a69a", "#eef3f8"
THA_FILL = "#e4e2da"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 8, "text.color": INK, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})
DPI = 600


def num(x):
    return f"{x:,.0f}" if abs(x) >= 10000 else f"{x:.0f}"


def scalebar(ax, x0, y0, km, fs=7):
    ax.plot([x0, x0 + km * 1000], [y0, y0], color=INK, lw=2, solid_capstyle="butt", zorder=5)
    ax.text(x0 + km * 500, y0 + km * 120, f"{km} km", ha="center", va="bottom", fontsize=fs, color=INK2)


def north(ax, x, y, d=30000):
    ax.annotate("N", xy=(x, y), xytext=(x, y - d), ha="center", va="center", fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))


bnd = gpd.read_file(BND)
tha = gpd.read_file(THA1).to_crs(32647)
with zipfile.ZipFile(NE) as z:
    ne = gpd.read_file(f"zip://{NE}!ne_10m_admin_0_countries.shp")
from shapely.geometry import box
ne = gpd.clip(ne, box(88, -2, 112, 26)).to_crs(32647)  # clip first: whole-world polygons smear in UTM

fig = plt.figure(figsize=(9.0, 6.0))
axL = fig.add_axes([0.0, 0.0, 0.34, 1.0])
axR = fig.add_axes([0.37, 0.0, 0.63, 1.0])

# ---------------------------------------------------------------- left: Thailand locator
tx0, ty0, tx1, ty1 = tha.total_bounds
pad = 60000
axL.set_xlim(tx0 - pad, tx1 + pad); axL.set_ylim(ty0 - pad, ty1 + pad)
axL.set_facecolor(SEA)
ne[ne.ADM0_A3 != "THA"].plot(ax=axL, color="#ffffff", edgecolor="#c3c2b7", lw=0.4)
tha.plot(ax=axL, color=THA_FILL, edgecolor="#b5b3a8", lw=0.25)
bnd.plot(ax=axL, color=STUDY, edgecolor="#f7f6f2", lw=0.3, zorder=3)
bnd.dissolve().boundary.plot(ax=axL, color=INK, lw=0.6, zorder=3)
for name, lon, lat in (("MYANMAR", 97.55, 16.3), ("LAO PDR", 102.6, 19.6), ("CAMBODIA", 104.6, 12.9),
                       ("MALAYSIA", 101.8, 5.4), ("THAILAND", 101.6, 15.2)):
    x, y = pyproj.Transformer.from_crs(4326, 32647, always_xy=True).transform(lon, lat)
    axL.text(x, y, name, ha="center", va="center", fontsize=6.5 if name != "THAILAND" else 7.5,
             color=MUTED if name != "THAILAND" else INK2, fontweight="bold" if name == "THAILAND" else "normal")
bx0, by0, bx1, by1 = bnd.total_bounds
axL.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=INK, lw=0.9, zorder=4))
scalebar(axL, tx0 - pad + 40000, ty0 - pad + 50000, 200, fs=6.5)
# one north arrow only (panel b), user 2026-10-05
axL.set_aspect("equal"); axL.set_xticks([]); axL.set_yticks([])
for s in axL.spines.values():
    s.set_edgecolor(INK2); s.set_linewidth(0.6)

# ---------------------------------------------------------------- right: study area (content of the earlier single-panel Fig 1)
pres = pd.read_csv(PRES)
ax = axR
bnd.plot(ax=ax, color=LAND, edgecolor=INK2, linewidth=0.6)
for src, c, z in (("factory", "#2a78d6", 3), ("confirmed", "#eb6834", 4)):
    s = pres[pres.source == src]
    ax.scatter(s.easting, s.northing, s=3, color=c, lw=0, alpha=0.75, zorder=z)
ck = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True).transform(100.0894, 20.3622)
ax.scatter(*ck, marker="^", s=55, color=INK, zorder=6)
for _, r in bnd.iterrows():
    p = r.geometry.representative_point()
    ax.text(p.x, p.y, r.ADM1_EN, ha="center", va="center", fontsize=7, color=INK2,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=7)
scalebar(ax, bx0 + 5000, by0 + 12000, 50)
north(ax, bx1 - 15000, by1 - 5000)
ax.set_aspect("equal"); ax.set_axis_off()
hd = [Line2D([], [], ls="", marker="o", ms=5, color="#2a78d6", label=f"Not individually checked (n = {num((pres.source == 'factory').sum())})"),
      Line2D([], [], ls="", marker="o", ms=5, color="#eb6834", label=f"Visually confirmed (n = {num((pres.source == 'confirmed').sum())})"),
      Line2D([], [], ls="", marker="^", ms=7, color=INK, label="Chiang Khong border crossing")]
ax.legend(handles=hd, loc="upper left", fontsize=7, handletextpad=0.3, title="Industrial firm locations",
          title_fontsize=7.5, alignment="left", frameon=False)

# ---------------------------------------------------------------- zoom connectors (box corners -> panel corners)
rx0, rx1 = ax.get_xlim(); ry0, ry1 = ax.get_ylim()
for (xa, ya), (xb, yb) in (((bx1, by1), (rx0, ry1)), ((bx1, by0), (rx0, ry0))):
    fig.add_artist(ConnectionPatch(xyA=(xa, ya), coordsA=axL.transData, xyB=(xb, yb), coordsB=ax.transData,
                                   color=MUTED, lw=0.6, ls=(0, (3, 2))))
ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False, ec=INK2, lw=0.6, zorder=0, clip_on=False))
axL.text(0.97, 0.015, "(a)", transform=axL.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")
ax.text(0.985, 0.015, "(b)", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_study_area.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.04)
print("written fig_study_area.png/.pdf")
