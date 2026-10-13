# Pre-registration — Time-disjoint out-of-sample evaluation

Written **before** the pre-2013 data was fetched or any number computed. This is a second,
independent confirmatory test. It complements `PREREGISTRATION.md` (ticker-disjoint but
time-overlapping) by holding the **universe fixed and holding out the time period**.

## Motivation

The first holdout changed the tickers but kept 2013–2024. A reviewer could object that the
result is regime-specific. This test freezes the **original 20-stock universe** and
evaluates on an **earlier, unseen era**.

## Held-out period and universe

- Universe: the **original 20** cross-sector large-caps (unchanged).
- Period: **2005-01-01 → 2012-12-31** (~8 years, includes the 2008 crisis). Never fetched
  or examined before this run.
- Data cached separately (`data/cache_pretime/`), adjusted close.
- Survivorship bias applies (fixed survivors) and is acknowledged.

## Frozen specification (identical to the prior tests; no tuning)

- Symbolization: `quantile_scope=pooled`, cutoffs 20/40/60/80, `market_neutral=True`
  (`pooled_mnl`), with `per_stock` as the context arm.
- Clustering: average linkage, k by silhouette over [2, min(6, ⌊N/3⌋)], min cluster size 3.
- Distance: banded Needleman-Wunsch, graded matrix, affine gaps, band 5.
- Allocator: two-stage, risk-parity across clusters, inverse-vol within, cap `max(10%, 1.5/N)`.
- Window 252, rebalance 21 days, cost 10 bps on Σ|Δw|.
- **Null gate bypassed** (frozen; passes ~90%).

## Hypothesis and decision rule

- **H2:** on the held-out period, `pooled_mnl` has a higher Sharpe than equal weight.
- **Primary test:** paired block bootstrap (block 20, 5,000 resamples, 95% CI) on the
  Sharpe difference `pooled_mnl − equal_weight`.
- **H2 supported** iff the CI excludes 0 and is positive. Otherwise not supported.
- One shot; no re-running with tweaked parameters.

## Interpretation guide

- If H2 is also not supported, the conclusion strengthens: the method does not beat equal
  weight on either a fresh universe or a fresh period — the pipeline is sound but the
  strategy has no reliable edge.
- If H2 is supported, it would be one positive result among several nulls and would still
  need replication before any claim.
