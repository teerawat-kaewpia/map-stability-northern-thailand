"""POST-HOC consensus core / configuration-dependent hotspot areas (Paper D).

Implements posthoc_consensus_spec_2026-10-02.md exactly. Reads the saved map-stability prediction arrays
(read-only), fits nothing, writes only into this folder. Refuses to run if the spec hash, any input hash,
or the recovered province-code order does not check out.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "exp_map_stability"
MAPS = EXP / "02_data" / "maps"
SPEC = HERE / "posthoc_consensus_spec_2026-10-02.md"
SPEC_SHA = (HERE / "posthoc_consensus_spec_2026-10-02.sha256").read_text().split()[0]
MANIFEST = EXP / "04_notes" / "results_manifest_2026-10-01.json"
SIM = EXP / "02_data" / "runs" / "map_similarity_by_province.csv"
TP_MANIFEST = EXP / "02_data" / "test_points" / "test_points_manifest.csv"

RADII = ["1km", "5km", "10km", "20km", "25km", "Rinf"]
LEARNERS = ["RF", "XGB", "LR"]
SEEDS = [42, 0, 1, 7, 2024]
CORE, CELL_KM2 = 0.9, 0.25


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def manifest_hashes():
    out = {}
    def walk(o):
        if isinstance(o, dict):
            if "path" in o and "sha256" in o:
                out[o["path"].replace("\\", "/")] = o["sha256"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(json.loads(MANIFEST.read_text(encoding="utf-8")))
    return out


def main():
    assert sha(SPEC) == SPEC_SHA, "spec changed after its hash was recorded"
    mh = manifest_hashes()
    names = [f"{L}_{r}_s{s}.npy" for r in RADII for L in LEARNERS for s in SEEDS] + \
            ["valid_mask.npy", "province_raster.npy"]
    for n in names:
        assert sha(MAPS / n) == mh[f"02_data/maps/{n}"], f"input hash mismatch: {n}"
    input_hashes = {n: mh[f"02_data/maps/{n}"] for n in names}

    valid = np.load(MAPS / "valid_mask.npy")
    prov = np.load(MAPS / "province_raster.npy")
    pv = prov[valid]                                              # province index per valid cell
    nv = int(valid.sum())

    # I[r, l, s, cell] - study-wide top 10 % per map (locked rule U7)
    I = np.zeros((len(RADII), len(LEARNERS), len(SEEDS), nv), dtype=np.uint8)
    for i, r in enumerate(RADII):
        for j, L in enumerate(LEARNERS):
            for k, s in enumerate(SEEDS):
                v = np.load(MAPS / f"{L}_{r}_s{s}.npy")
                assert v.shape == (nv,)
                I[i, j, k] = v >= np.quantile(v, 0.9)

    # recover province code order from the saved hotspot counts (spec: must match exactly)
    sim = pd.read_csv(SIM)
    ref = sim[(sim.seed == 42) & (sim.a == "RF_1km") & (sim.b == "RF_10km") & (sim.metric == "jaccard_top10")]
    ref_a = dict(zip(ref.province, ref.n_top10_a.astype(int)))
    ref_b = dict(zip(ref.province, ref.n_top10_b.astype(int)))
    ia, ib = I[0, 0, 0], I[2, 0, 0]                               # RF_1km_s42, RF_10km_s42
    codes = {}
    for idx in sorted(set(pv.tolist())):
        ca, cb = int(ia[pv == idx].sum()), int(ib[pv == idx].sum())
        hit = [c for c in ref_a if ref_a[c] == ca and ref_b[c] == cb]
        assert len(hit) == 1, f"province index {idx}: ambiguous/no match {hit}"
        codes[idx] = hit[0]
    assert sorted(codes.values()) == sorted(ref_a), "province order not fully recovered"
    tp = pd.read_csv(TP_MANIFEST, encoding="utf-8")
    pname = dict(zip(tp.ADM1_PCODE, tp.province_en))

    rows, ssrows, dist = [], [], []
    freqs = {}
    for set_name, ridx in [("A_all90", list(range(6))), ("B_no1km_75", list(range(1, 6)))]:
        Is = I[ridx].astype(np.float32)                           # (R, L, S, N)
        f = Is.mean(axis=(0, 1, 2))
        freqs[set_name] = f
        np.save(HERE / f"consensus_freq_set{set_name[0]}.npy", f)
        nmaps = Is.shape[0] * Is.shape[1] * Is.shape[2]
        # descriptive per-cell SS decomposition (balanced design)
        m = f
        R, Lc, S = Is.shape[:3]
        ss_tot = ((Is - m) ** 2).sum(axis=(0, 1, 2))
        ss_r = Lc * S * ((Is.mean(axis=(1, 2)) - m) ** 2).sum(axis=0)
        ss_l = R * S * ((Is.mean(axis=(0, 2)) - m) ** 2).sum(axis=0)
        ss_s = R * Lc * ((Is.mean(axis=(0, 1)) - m) ** 2).sum(axis=0)
        cls = np.where(f >= CORE, "core", np.where(f > 0, "dependent", "never"))
        for kcount in range(nmaps + 1):
            dist.append({"set": set_name, "n_maps_hotspot": kcount, "f": kcount / nmaps,
                         "cells": int((np.rint(f * nmaps) == kcount).sum())})
        groups = [("ALL", "Corridor (9 provinces)", np.ones(nv, bool))] + \
                 [(codes[i], pname[codes[i]], pv == i) for i in sorted(codes)]
        for code, name, g in groups:
            rows.append({"set": set_name, "province": code, "name": name, "valid_km2": g.sum() * CELL_KM2,
                         "core_km2": ((cls == "core") & g).sum() * CELL_KM2,
                         "dependent_km2": ((cls == "dependent") & g).sum() * CELL_KM2,
                         "never_km2": ((cls == "never") & g).sum() * CELL_KM2,
                         "ever_hotspot_km2": ((f > 0) & g).sum() * CELL_KM2,
                         "core_share_of_ever": ((cls == "core") & g).sum() / max(((f > 0) & g).sum(), 1)})
            d = (cls == "dependent") & g
            T = ss_tot[d].sum()
            ssrows.append({"set": set_name, "province": code, "name": name, "dependent_cells": int(d.sum()),
                           "share_radius": ss_r[d].sum() / T if T else np.nan,
                           "share_learner": ss_l[d].sum() / T if T else np.nan,
                           "share_seed": ss_s[d].sum() / T if T else np.nan,
                           "share_interactions": (T - ss_r[d].sum() - ss_l[d].sum() - ss_s[d].sum()) / T if T else np.nan})
    pd.DataFrame(rows).to_csv(HERE / "consensus_by_province.csv", index=False)
    pd.DataFrame(ssrows).to_csv(HERE / "ss_decomposition_by_province.csv", index=False)
    pd.DataFrame(dist).to_csv(HERE / "f_distribution.csv", index=False)

    # figure: f (Set A) and classes for A and B
    H, W = valid.shape
    def grid(v):
        a = np.full((H, W), np.nan, np.float32); a[valid] = v; return a
    fig, ax = plt.subplots(1, 3, figsize=(15, 7.2), constrained_layout=True)
    im = ax[0].imshow(grid(freqs["A_all90"]), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=ax[0], shrink=0.6, label="share of maps with cell in top 10%")
    ax[0].set_title("(a) Hotspot frequency, Set A (90 maps)")
    cmap = ListedColormap(["#e6e6e6", "#f0a830", "#1b5e8c"])
    for a_, (sn, ttl) in zip(ax[1:], [("A_all90", "(b) Classes, Set A (90 maps)"),
                                      ("B_no1km_75", "(c) Classes, Set B (75 maps, no 1 km)")]):
        fq = freqs[sn]
        c = np.where(fq >= CORE, 2, np.where(fq > 0, 1, 0)).astype(np.float32)
        a_.imshow(grid(c), cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        a_.set_title(ttl)
    for a_ in ax:
        a_.contour(np.where(prov > 0, prov, np.nan), levels=np.arange(1.5, 10), colors="k", linewidths=0.3)
        a_.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(3)]
    fig.legend(handles, ["never hotspot (f = 0)", "configuration-dependent (0 < f < 0.9)",
                         "consensus core (f >= 0.9)"], loc="lower center", ncol=3, frameon=False)
    fig.suptitle("POST-HOC: consensus of top-10% hotspots across radius x learner x seed (500 m grid)")
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig_consensus_map.{ext}", dpi=300)

    outs = ["consensus_freq_setA.npy", "consensus_freq_setB.npy", "consensus_by_province.csv",
            "ss_decomposition_by_province.csv", "f_distribution.csv", "fig_consensus_map.png", "fig_consensus_map.pdf"]
    (HERE / "outputs_sha256.json").write_text(json.dumps({
        "spec_sha256": SPEC_SHA, "script_sha256": sha(__file__), "province_order": codes,
        "inputs": input_hashes, "outputs": {o: sha(HERE / o) for o in outs}}, indent=2, default=str))
    print("province order", codes)
    print(pd.DataFrame(rows).query("province == 'ALL'").to_string())
    print(pd.DataFrame(ssrows).query("province == 'ALL'").to_string())


if __name__ == "__main__":
    main()
