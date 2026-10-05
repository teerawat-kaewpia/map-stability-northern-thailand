<!-- extracted from predeclaration_2026-10-01_redacted.md -->
### Amendment 3 - 2026-10-01, before any run and before the test points exist

**Why.** Item 3 above ("if the 1 km ring cannot yield the required number ... not computable") had no
stopping point for the training-background draw, and the number of training background points in a
leave-one-province-out fold was not fixed. Without both, whether a configuration is "computable" could
be decided after seeing it. Amendments 1-2 are left unchanged.

**A3.1 Training background size.** In each fold `n_bg = n_presence_train = 2,859 - n_presence_test`
(1:1, as in Paper C). Models fitted on all nine provinces to produce the maps use `n_bg = 2,859`.

| held-out province | test presences | `n_bg` (training) | round-1 draw | early-stop at |
|---|---|---|---|---|
| Chiang Mai | 906 | 1,953 | 39,060 | 5,859 |
| Chiang Rai | 559 | 2,300 | 46,000 | 6,900 |
| Lampang | 468 | 2,391 | 47,820 | 7,173 |
| Lamphun | 332 | 2,527 | 50,540 | 7,581 |
| Uttaradit | 201 | 2,658 | 53,160 | 7,974 |
| Nan | 146 | 2,713 | 54,260 | 8,139 |
| Phrae | 113 | 2,746 | 54,920 | 8,238 |
| Phayao | 81 | 2,778 | 55,560 | 8,334 |
| Mae Hong Son | 53 | 2,806 | 56,120 | 8,418 |
| all nine (map models) | - | 2,859 | 57,180 | 8,577 |

**A3.2 Training-sampling budget - Paper C's rule, every radius and every seed.** The rule is the one
implemented in `paper_C/03_results/scripts/paperC_spatial_transfer_lopo.py`, lines 121-148 (file
hashed in the input lock):
- One generator `numpy.random.default_rng(seed)` is created at the start of each draw and continues
  across rounds.
- At most **6 rounds**. Round *i* (1-6) draws `k_i = i x max(20 x n_bg, 20,000)` points uniformly in the
  bounding box of the sampling polygon - eastings first, then northings, as in that code. The sampling
  polygon is the union of the training provinces in a fold, or the nine-province polygon for map
  models.
- A point is kept if it lies inside the sampling polygon, at least 500 m from the nearest **training**
  presence, and within the outer radius from it. Kept points accumulate in draw order. Drawing stops
  after round *i* once at least `3 x n_bg` points have been kept.
- Covariates are then extracted for all kept points; points with any missing covariate are dropped;
  the first `n_bg` remaining, in draw order, are used.
- If fewer than `n_bg` remain, that fold / configuration / seed is recorded as **"incomplete under the
  locked training-sampling budget"**. No round is added and `n_bg` is not changed.

Distances for the ring and the 500 m exclusion are measured to **training presences only**, as in
Paper C. A training background point can therefore lie within 500 m of a test presence that sits
outside its own province polygon (the exceptions listed in A2.1).

**A3.3 Paired comparison.** Within a fold and a seed, the draw for every radius starts a fresh
generator from that same seed. Because `n_bg` - and therefore every `k_i` - is the same for all radii in
a fold, all radii see the identical candidate sequence; they differ only in the ring filter and, as a
consequence, possibly in how many rounds are drawn before the stop. The background drawn for a fold,
radius and seed is generated once and **shared by all three learners**, so learner comparisons within
a radius are paired as well.

**A3.4 `R-inf`.** Same rule with only the outer-radius condition removed (the inner 500 m exclusion and
the polygon remain). Paper C's function takes the outer radius as a number, so this is implemented in
the new runner; the old function is not called with `None`.

**A3.5 What the earlier evidence does and does not show.** In Objective_One a 1 km ring yielded 822 of
2,868 points under a single draw of 20 x n. That motivated locking a budget; it is **not** evidence
that the 1 km ring will be incomplete under the six-round rule above, and no expectation about that
outcome is recorded here.

**Not changed by this amendment:** everything in Amendments 1-2 and in the original text other than
the stopping rule and sample size of the training background.
