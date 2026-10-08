# GenomeFolio — Follow-up Experiments

Run: `uv run python scripts/run_experiments.py` (200 gate draws, 10 bps, same 906-day
report period). Raw tables: `reports/experiments.csv`, `reports/experiments_ci.csv`,
`reports/experiments.log`.

Answers two questions from `reports/RESULTS.md`:

1. **Concentration hypothesis** — is the underperformance just equal weight *across
   clusters* over-weighting small clusters?
2. **Isolate the alignment step** — does a plain correlation distance with the *same*
   clustering and allocator do better?

The null gate is computed once per rebalance and the clusters reused across all portfolio
variants, so the sweep is cheap and the comparison is clean.

## Bug found and fixed along the way

The first run showed `align_risparity_equal` **byte-identical** to `align_equal_equal`. In
`_cluster_first_stage` the guard `if cluster_method == "equal" or forecasts is None` made
any no-forecast call fall through to equal, so **`risk_parity` was silently ignored**
whenever forecasts were absent (which is every baseline path and this sweep). Fixed in
`src/portfolio.py`; regression tests added (`test_risk_parity_honored_without_forecasts`,
`test_forecast_tilt_without_forecasts_falls_back_to_equal`). The numbers below are the
corrected run.

## Results (200 draws, 10 bps)

| variant | Sharpe | ann_return | ann_vol | max_dd | turnover |
|---|---|---|---|---|---|
| equal_weight | 0.746 | 0.0944 | 0.133 | -0.157 | 0.032 |
| align_equal_equal | 0.332 | 0.0359 | 0.133 | -0.179 | 0.299 |
| align_risparity_equal | 0.336 | 0.0362 | 0.132 | -0.180 | 0.269 |
| align_equal_invvol | 0.367 | 0.0395 | 0.128 | -0.172 | 0.319 |
| align_risparity_invvol | 0.379 | 0.0406 | 0.126 | -0.173 | 0.272 |
| align_tilt_invvol | 0.476 | 0.0537 | 0.127 | -0.172 | 0.327 |
| corr_equal_equal | 0.506 | 0.0609 | 0.135 | -0.159 | 0.170 |
| corr_risparity_invvol | 0.530 | 0.0615 | 0.128 | -0.155 | 0.171 |

## Paired 95% CI on Sharpe difference vs equal_weight

| variant | Sharpe diff vs equal_weight |
|---|---|
| align_equal_equal | (-0.688, -0.193) — worse |
| align_risparity_equal | (-0.662, -0.211) — worse |
| align_equal_invvol | (-0.673, -0.127) — worse |
| align_risparity_invvol | (-0.657, -0.128) — worse |
| align_tilt_invvol | (-0.573, **-0.006**) — marginally worse |
| corr_equal_equal | (-0.599, **0.074**) — indistinguishable |
| corr_risparity_invvol | (-0.626, **0.168**) — indistinguishable |

## What this says

**1. Concentration is real but small — not the explanation.** Moving from equal/equal to
risk-parity across clusters + inverse-vol within lifts Sharpe only 0.332 → 0.379 (+0.047)
and cuts turnover 0.299 → 0.272. It is still far below equal weight (0.746) and the CI
still excludes 0. So over-weighting small clusters explains a sliver, not the gap.

**2. The alignment distance is the weak link.** With the *same* constrained-k clustering
and the *same* allocator, correlation distance (`1 - corr`) reaches **0.506–0.530** versus
**0.332–0.379** for alignment — and correlation clustering is **statistically
indistinguishable from equal weight**. The clustering machinery and the allocator are
fine; the sequence-alignment similarity is what destroys performance. Alignment clusters
also churn more (turnover ~0.27–0.33 vs ~0.17).

**3. The forecast tilt is the best alignment variant** (0.476), closing most of the gap,
but its CI upper bound is a hair below 0 — still marginally worse than equal weight.
Treat with caution: only 31 rebalances, and the tilt is fit on the same window.

## Caveats

- 200 gate draws here (the sweep budget) versus 1000 in `RESULTS.md`; every variant shares
  the same clusters, so the **comparisons** are unaffected even if the absolute gate
  pass rate shifts slightly.
- Single report period, fixed universe, ~31 rebalances → wide CIs.
- `corr_*` uses Pearson correlation over the training window; it is a different similarity,
  not a strict "alignment vs nothing" test, but it is the fairest same-machinery control.

## Next

The alignment step itself now looks like the problem. Worth checking whether the
per-stock quantile symbolization (which discards magnitude) is throwing away the signal —
i.e. re-run the `market_neutral` and `15/35/65/85` ablations, and consider an
alignment-with-magnitude variant.
