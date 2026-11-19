"""P2 summary (spec Note 1): AUC change from the submitted values, per province and for the manuscript summary
(median over the five draws of the median over the nine folds), for the reassigned and the excluded variant."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
STORED = HERE.parents[2] / "exp_map_stability" / "02_data" / "runs" / "lopo_metrics_by_fold.csv"
RADII = ["1km", "5km", "10km", "20km", "25km", "Rinf"]

rows = []
for variant in ["reassign", "exclude"]:
    files = sorted((OUT / f"P2_{variant}").glob("*.json"))
    assert len(files) == 270, (variant, len(files))
    for f in files:
        u = json.loads(f.read_text(encoding="utf-8"))
        assert u["status"] == "complete", f.name
        for lr, auc in u["auc"].items():
            rows.append(dict(variant=variant, fold=u["fold"], radius=u["radius"], seed=u["seed"], learner=lr,
                             n_test_presence=u["n_test_presence"], auc=auc))
new = pd.DataFrame(rows)
old = pd.read_csv(STORED, encoding="utf-8-sig")[["fold", "radius", "seed", "learner", "auc"]]
old = old.rename(columns={"auc": "auc_submitted"})
m = new.merge(old, on=["fold", "radius", "seed", "learner"], how="left", validate="many_to_one")
assert m["auc_submitted"].notna().all()
m["d_auc"] = m["auc"] - m["auc_submitted"]
m.to_csv(OUT / "P2_auc_by_unit.csv", index=False)

prov = (m.groupby(["variant", "fold"])
        .agg(n_test_presence=("n_test_presence", "first"), median_d_auc=("d_auc", "median"),
             min_d_auc=("d_auc", "min"), max_d_auc=("d_auc", "max")).reset_index())
prov.to_csv(OUT / "P2_delta_by_province.csv", index=False)


def summary(df, col):
    s = df.groupby(["learner", "radius", "seed"])[col].median().groupby(["learner", "radius"]).median()
    return s


parts = []
for variant, g in m.groupby("variant"):
    s = pd.DataFrame({"auc": summary(g, "auc"), "auc_submitted": summary(g, "auc_submitted")}).reset_index()
    s["d_auc"] = s["auc"] - s["auc_submitted"]
    s.insert(0, "variant", variant)
    parts.append(s)
summ = pd.concat(parts)
summ["radius"] = pd.Categorical(summ["radius"], RADII, ordered=True)
summ = summ.sort_values(["variant", "learner", "radius"])
summ.to_csv(OUT / "P2_auc_summary.csv", index=False)

pd.set_option("display.width", 160)
print(prov.round(4).to_string(index=False))
print(summ.pivot_table(index=["variant", "learner"], columns="radius", values="auc", observed=True).round(3))
print(summ.pivot_table(index=["variant", "learner"], columns="radius", values="d_auc", observed=True).round(4))
print("max |summary d_auc| by variant:", summ.groupby("variant")["d_auc"].apply(lambda x: x.abs().max()).round(4).to_dict())
