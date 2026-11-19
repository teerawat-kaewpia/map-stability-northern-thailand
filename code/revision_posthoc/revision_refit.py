"""Revision post hoc analyses that refit models: P1 (near-firm test backgrounds), P5 (SHAP for all draws),
P2 (six mismatched presences). spec_revision_posthoc_2026-11-01.md, hashed before computing.
Uses the locked runner's functions (imported, not edited). Resumable: one JSON per unit under ../outputs/.
Usage: python revision_refit.py P1|P5|P2"""
import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.ops import unary_union
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
EXP = HERE.parents[1] / "exp_map_stability"
sys.path.insert(0, str(EXP / "01_scripts"))
import run_map_stability as R  # noqa: E402

SPEC_SHA = "7434b645295c28cffc7d33e4eff1ba8642353017a3a21da7808d71ace7c9082a"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / "spec_revision_posthoc_2026-11-01_v1_as_hashed.md") == SPEC_SHA, "spec changed after hashing"
LOCK = json.loads(R.LOCK.read_text(encoding="utf-8"))
R.verify(LOCK, "dry")                                        # every locked input, test point and the predeclaration
D = R.Data(LOCK)
LOPO = R.RUNS / "lopo"
STORED = pd.read_csv(R.RUNS / "lopo_metrics_by_fold.csv", encoding="utf-8-sig")
log = R.log


def write_json(p, d):
    tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(d, indent=1), encoding="utf-8"); tmp.replace(p)


def test_set(pres_te, tb):
    return pd.concat([pres_te[["easting", "northing"] + R.FEATURES].assign(presence=1),
                      tb[["easting", "northing"] + R.FEATURES].assign(presence=0)], ignore_index=True)


# ================================================================= P1
def near_test_points(Dmax):
    """make_test_points.py procedure with an upper distance bound; rng 20261101 per province."""
    f = OUT / f"P1_test_points_near{int(Dmax / 1000)}km.csv"
    if f.exists():
        return pd.read_csv(f, encoding="utf-8-sig")
    tree = cKDTree(D.pres[["easting", "northing"]].to_numpy()); parts, man = [], []
    for held in D.provs:
        code, poly = D.code[held], D.poly[held]
        n_test = int((D.pres["FPROVNAME"] == held).sum()); B = max(200000, 2000 * n_test)
        rng = np.random.default_rng(20261101); minx, miny, maxx, maxy = poly.bounds
        e = rng.uniform(minx, maxx, B); n = rng.uniform(miny, maxy, B)
        dmin, _ = tree.query(np.column_stack([e, n]), k=1)
        band = (dmin >= R.INNER_M) & (dmin <= Dmax)
        idx = np.flatnonzero(band)
        inside = gpd.GeoSeries(gpd.points_from_xy(e[idx], n[idx]), crs="EPSG:32647").within(poly).to_numpy()
        idx = idx[inside]
        C = pd.DataFrame({"ADM1_PCODE": code, "draw_index": idx, "easting": e[idx], "northing": n[idx],
                          "dist_nearest_presence_m": dmin[idx]})
        C = pd.concat([C, D.extract(C.easting.to_numpy(), C.northing.to_numpy())], axis=1)
        C = C[np.isfinite(C[R.FEATURES].to_numpy()).all(axis=1)].sort_values("draw_index").head(n_test)
        man.append({"ADM1_PCODE": code, "n_test": n_test, "rows": len(C), "complete": len(C) == n_test})
        parts.append(C)
    T = pd.concat(parts, ignore_index=True); T.to_csv(f, index=False, encoding="utf-8-sig")
    pd.DataFrame(man).to_csv(OUT / f"P1_test_points_near{int(Dmax / 1000)}km_manifest.csv", index=False)
    log(f"P1 test points near {Dmax / 1000:.0f} km: {len(T)} rows, incomplete: {[m['ADM1_PCODE'] for m in man if not m['complete']]}")
    return T


def run_p1():
    near = {lab: near_test_points(d) for lab, d in (("near1km", 1000.0), ("near5km", 5000.0))}
    od = OUT / "P1_units"; od.mkdir(exist_ok=True)
    for held in D.provs:
        code = D.code[held]
        pres_tr = D.pres[D.pres["FPROVNAME"] != held]; pres_te = D.pres[D.pres["FPROVNAME"] == held]
        tests = {"original": test_set(pres_te, D.test_bg[code])}
        for lab, T in near.items():
            tb = T[T.ADM1_PCODE == code]
            if len(tb) == len(pres_te):
                tests[lab] = test_set(pres_te, tb)
        for seed in R.SEEDS:
            for rlab, _ in R.RADII:
                f = od / f"{code}_{rlab}_s{seed}.json"
                if f.exists():
                    continue
                fbg = LOPO / f"{code}_{rlab}_s{seed}_bg.npy"
                if not fbg.exists():
                    write_json(f, {"fold": code, "radius": rlab, "seed": seed, "status": "no background file"}); continue
                t0 = time.time(); xy = np.load(fbg)
                bg = D.extract(xy[:, 0], xy[:, 1]); bg["easting"], bg["northing"] = xy[:, 0], xy[:, 1]
                train = pd.concat([pres_tr[["easting", "northing"] + R.FEATURES].assign(presence=1),
                                   bg[["easting", "northing"] + R.FEATURES].assign(presence=0)], ignore_index=True)
                rec = {"fold": code, "radius": rlab, "seed": seed, "status": "complete", "learners": {}}
                for L in R.LEARNERS:
                    m = R.make(L).fit(train[R.FEATURES], train["presence"])
                    res = {}
                    for lab, T in tests.items():
                        res[f"auc_{lab}"] = float(roc_auc_score(T["presence"], m.predict_proba(T[R.FEATURES])[:, 1]))
                    st = STORED[(STORED.fold == code) & (STORED.radius == rlab) & (STORED.seed == seed) & (STORED.learner == L)]
                    res["auc_stored"] = float(st.auc.iloc[0]) if len(st) else None
                    res["reproduced"] = bool(res["auc_stored"] is not None and abs(res["auc_original"] - res["auc_stored"]) <= 1e-9)
                    rec["learners"][L] = res
                rec["seconds"] = round(time.time() - t0, 1); write_json(f, rec)
                log(f"P1 {code} {rlab:<5} s{seed:<4} " + " ".join(
                    f"{L}:{v['auc_original']:.3f}/{v.get('auc_near5km', float('nan')):.3f}/{v.get('auc_near1km', float('nan')):.3f}"
                    + ("" if v["reproduced"] else "!") for L, v in rec["learners"].items()) + f" ({rec['seconds']}s)")


# ================================================================= P5
def run_p5():
    import shap, joblib
    G = R.build_grid(D); nv = len(G["X"])
    cells = np.load(R.RUNS / "shap" / "explained_cells.npy"); bgi = np.load(R.RUNS / "shap" / "background_cells.npy")
    Xc, Xb = G["X"].iloc[cells][R.FEATURES], G["X"].iloc[bgi][R.FEATURES]
    od = OUT / "P5_shap"; od.mkdir(exist_ok=True)
    all_xy = D.pres[["easting", "northing"]].to_numpy(); poly9 = unary_union(list(D.bnd.geometry))
    checks = []
    for seed in R.SEEDS:
        for rlab, outer in R.RADII:
            need = [L for L in R.LEARNERS if not (od / f"shap_{L}_{rlab}_s{seed}.npy").exists()]
            if not need:
                continue
            if seed == 42:                                    # seed-42 values on disk are reused (spec P5.2)
                for L in need:
                    np.save(od / f"shap_{L}_{rlab}_s42.npy", np.load(R.RUNS / "shap" / f"shap_{L}_{rlab}.npy"))
                continue
            bg, info = R.sample_bg(D, all_xy, poly9, 2859, outer, seed)
            train = pd.concat([D.pres[R.FEATURES].assign(presence=1), bg[R.FEATURES].assign(presence=0)], ignore_index=True)
            for L in need:
                m = R.make(L).fit(train[R.FEATURES], train["presence"])
                pred = m.predict_proba(G["X"][R.FEATURES])[:, 1].astype("float32")
                stored = np.load(R.MAPS / f"{L}_{rlab}_s{seed}.npy")
                dev = float(np.abs(pred - stored).max())
                checks.append({"learner": L, "radius": rlab, "seed": seed, "max_abs_diff": dev, "reproduced": dev <= 1e-6})
                if L == "LR":
                    fn = lambda X, m=m: m.predict_proba(pd.DataFrame(X, columns=R.FEATURES))[:, 1]
                    ex = shap.explainers.Permutation(fn, shap.maskers.Independent(Xb.to_numpy(), max_samples=100),
                                                     seed=R.SHAP_PERM_SEED)
                    sv = ex(Xc.to_numpy(), max_evals=R.SHAP_PERM_MAX_EVALS, silent=True).values
                else:
                    sv = shap.TreeExplainer(m, data=Xb, feature_perturbation="interventional",
                                            model_output="probability").shap_values(Xc)
                    sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv
                np.save(od / f"shap_{L}_{rlab}_s{seed}.npy", np.asarray(sv, dtype="float64"))
                log(f"P5 {L} {rlab:<5} s{seed:<4} map max|diff|={dev:.1e} SHAP done")
            pd.DataFrame(checks).to_csv(od / f"map_reproduction_s{seed}_{rlab}.csv", index=False)
    rows = []
    for seed in R.SEEDS:
        for rlab, _ in R.RADII:
            for L in R.LEARNERS:
                sv = np.load(od / f"shap_{L}_{rlab}_s{seed}.npy"); ma = np.abs(sv).mean(axis=0)
                rank = (-ma).argsort().argsort() + 1
                for j, f in enumerate(R.FEATURES):
                    rows.append({"learner": L, "radius": rlab, "seed": seed, "feature": f, "mean_abs_shap": float(ma[j]),
                                 "share": float(ma[j] / ma.sum()), "rank": int(rank[j])})
    pd.DataFrame(rows).to_csv(OUT / "P5_shap_all_draws.csv", index=False)
    log("P5 summary written")


# ================================================================= P2
def mismatched():
    pts = gpd.GeoSeries(gpd.points_from_xy(D.pres.easting, D.pres.northing), crs="EPSG:32647")
    poly_prov = []
    for p in pts:
        hit = [h for h in D.provs if D.poly[h].contains(p)]
        poly_prov.append(hit[0] if len(hit) == 1 else None)
    s = pd.Series(poly_prov, index=D.pres.index)
    mm = s.notna() & (s != D.pres["FPROVNAME"])
    return mm, s


def run_p2():
    mm, polyprov = mismatched()
    log(f"P2 mismatched presences: {int(mm.sum())}; outside every polygon: {int(polyprov.isna().sum())}")
    variants = {"reassign": D.pres.assign(FPROVNAME=np.where(mm, polyprov, D.pres["FPROVNAME"])),
                "exclude": D.pres[~mm].reset_index(drop=True)}
    for vname, P in variants.items():
        od = OUT / f"P2_{vname}"; od.mkdir(exist_ok=True)
        for held in D.provs:
            code = D.code[held]
            pres_tr = P[P["FPROVNAME"] != held]; pres_te = P[P["FPROVNAME"] == held]
            poly_tr = unary_union([D.poly[p] for p in D.provs if p != held])
            test = test_set(pres_te, D.test_bg[code]); n_bg = len(pres_tr)
            for seed in R.SEEDS:
                for rlab, outer in R.RADII:
                    f = od / f"{code}_{rlab}_s{seed}.json"
                    if f.exists():
                        continue
                    t0 = time.time()
                    bg, info = R.sample_bg(D, pres_tr[["easting", "northing"]].to_numpy(), poly_tr, n_bg, outer, seed)
                    rec = {"variant": vname, "fold": code, "radius": rlab, "seed": seed, "n_bg": n_bg,
                           "n_test_presence": len(pres_te), "sampling": info}
                    if bg is None:
                        rec["status"] = "incomplete"; write_json(f, rec); continue
                    train = pd.concat([pres_tr[["easting", "northing"] + R.FEATURES].assign(presence=1),
                                       bg[["easting", "northing"] + R.FEATURES].assign(presence=0)], ignore_index=True)
                    rec["status"], rec["auc"] = "complete", {}
                    for L in R.LEARNERS:
                        m = R.make(L).fit(train[R.FEATURES], train["presence"])
                        rec["auc"][L] = float(roc_auc_score(test["presence"], m.predict_proba(test[R.FEATURES])[:, 1]))
                    rec["seconds"] = round(time.time() - t0, 1); write_json(f, rec)
                    log(f"P2 {vname} {code} {rlab:<5} s{seed:<4} " + " ".join(f"{L}:{a:.3f}" for L, a in rec["auc"].items())
                        + f" ({rec['seconds']}s)")


if __name__ == "__main__":
    {"P1": run_p1, "P5": run_p5, "P2": run_p2}[sys.argv[1]]()
    log(f"{sys.argv[1]} complete")
