# Pre-registration — Holdout evaluation of the symbolization choice

Written **before** the holdout data was fetched or any holdout number was computed.

## Motivation

The symbolization ablations (`reports/ABLATIONS.md`) scanned six variants on the same
report period and the ranking did **not** replicate on the tune period. Scanning variants
and reporting the best is selection. This is a single, pre-registered confirmatory test on
data not used to make the choice.

## Held-out universe

A **new 20-stock US large-cap universe, disjoint from the original 20**, frozen here:

```
NVDA AMD QCOM TXN ORCL ADBE CRM IBM ABT BMY LLY AMGN GILD TGT LOW SBUX BA GE MMM HON
```

Period 2013-01-01 → 2025-01-01. Data cached separately (`data/cache_holdout/`), adjusted
close. Survivorship bias applies and is acknowledged (fixed universe of survivors).

**Out-of-sample-ness comes from the tickers:** none of these were used to choose the
symbolization, cluster, allocator, or any prior parameter. The time span overlaps the
original study, so macro-regime similarity is a limitation (stated, not hidden).

## Frozen specification (no tuning on the holdout)

- Symbolization: `quantile_scope=pooled`, cutoffs 20/40/60/80, `market_neutral=True`
  (the "`pooled_mnl`" arm).
- Clustering: hierarchical average linkage, k by silhouette over [2, min(6, ⌊N/3⌋)],
  min cluster size 3.
- Distance: banded Needleman-Wunsch, graded matrix, affine gaps, band w = 5.
- Allocator: two-stage, risk-parity across clusters, inverse-vol within,
  cap = max(10%, 1.5/N).
- Window 252, rebalance 21 days, transaction cost 10 bps on Σ|Δw|.
- **Null gate bypassed** (passes ~90%; same as the ablation study). This is frozen, not
  chosen post hoc.

## Hypothesis and endpoints

- **H1:** on the held-out universe, `pooled_mnl` has a **higher Sharpe** than equal weight.
- **Primary endpoint:** net (10 bps) annualized Sharpe over the full holdout walk-forward.
- **Primary test:** paired block bootstrap (block 20 days, 5,000 resamples, 95% CI) on the
  Sharpe difference `pooled_mnl − equal_weight`.
- **Secondary:** annualized-return difference, same bootstrap.
- **Context arms (not confirmatory):** `per_stock` (baseline symbolization) and equal
  weight, reported for comparison only.

## Decision rule (fixed now)

- **H1 supported** iff the 95% CI for the Sharpe difference excludes 0 **and** is positive.
- **H1 not supported** otherwise (including a CI that excludes 0 and is negative → the
  magnitude-aware claim fails).
- No other comparison is confirmatory. No re-running with tweaked parameters: this is one
  shot on this universe.

## What a null/negative result would mean

If H1 is not supported, the defensible conclusion is that no symbolization reliably beats
equal weight on this evidence, and the earlier report-period gains were selection — which
would be reported as-is.
