<!-- extracted from predeclaration_2026-10-01_redacted.md -->
### Amendment 2 - 2026-10-01, before any run and before the test points exist

**Why.** A review of Amendment 1 (no run, no result, no test points generated) found that one sentence
in A1.3 overstates the separation achieved by a registry-based split, and that the infeasibility
rule in A1.4 has no stopping point. Amendment 1 is left unchanged; where they conflict, this
amendment governs.

**A2.1 What the leave-one-province-out split does and does not separate (corrects A1.3).** The
sentence in A1.3 "The held-out province contributes nothing to fitting" is **withdrawn as worded**.
Folds are defined by registry province (`FPROVNAME`, A1.2), so what holds is: **no presence whose
registry province is the held-out province is in the training set.** It is **not** a strictly
geographic holdout. A spatial join of the 2,859 clean presences with the locked province polygons
(run 2026-10-01) finds six presences whose registry province differs from the polygon they lie in:

| FID | registry province | lies in polygon of | distance to that polygon's edge |
|---|---|---|---|
| [registry ID 1 redacted] | Lamphun | Chiang Mai | 19,823 m |
| [registry ID 2 redacted] | Lamphun | Chiang Mai | 17,275 m |
| [registry ID 3 redacted] | Chiang Mai | Lamphun | 926 m |
| [registry ID 4 redacted] | Chiang Mai | Lamphun | 1,958 m |
| [registry ID 5 redacted] | Chiang Rai | Phayao | 75 m |
| [registry ID 6 redacted] | Phrae | Nan | 5,345 m |

This creates exceptions in **six folds, in two directions**:
- *training presence located inside the held-out polygon*: Chiang Mai fold (2), Lamphun fold (2),
  Nan fold (1), Phayao fold (1);
- *test presence located outside the held-out polygon, while that fold's test background is drawn
  inside it (A1.4)*: Chiang Mai fold (2), Lamphun fold (2), Phrae fold (1), Chiang Rai fold (1).

These records are kept as they are: no presence is reassigned or removed, so that the folds match
Paper C. The six FIDs and the affected folds are reported with the per-fold results. Training
**background** remains confined to the training provinces' polygons at every radius, `R-inf`
included (A1.3 unchanged on this point; it is also how Paper C's runner passes the training-province
polygon to the sampler, `paperC_spatial_transfer_lopo.py` line 276).

**A2.2 Sampling budget for the test points (completes A1.4).** For each held-out province at most
**max(20,000, 200 x n_test)** candidates are drawn, where a candidate is one uniform draw in the
province's bounding box, counted before any filter. The first `n_test` candidates that pass all
A1.4 filters, in draw order, are used. If fewer than `n_test` pass within the budget, that fold is
reported as **"incomplete under the locked sampling budget"** - not as evidence that the province has
no usable points - and the budget is not raised after the fact. Budgets: Chiang Mai 181,200;
Chiang Rai 111,800; Lampang 93,600; Lamphun 66,400; Uttaradit 40,200; Nan 29,200; Phrae 22,600;
Phayao 20,000; Mae Hong Son 20,000.

**Not changed by this amendment:** everything in Amendment 1 other than the withdrawn sentence of
A1.3 and the stopping rule of A1.4.
