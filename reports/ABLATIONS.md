# GenomeFolio — Symbolization Ablations

Run: `uv run python scripts/run_symbolize_ablations.py [report|tune|all]`.
Raw tables: `reports/ablations_report.csv`, `ablations_tune.csv` (+ `_ci_`, `_stability_`),
logs `reports/ablations_*.log`.

Question: the alignment distance looked like the weak link in `EXPERIMENTS.md`, and the
symbolization throws away magnitude. Does changing the symbolization — especially keeping
magnitude — recover the signal?

All variants use the best allocator from the follow-up experiments (risk-parity across
clusters, inverse-vol within), 10 bps cost. **The null gate is bypassed** (it passes ~90%,
so clusters are used every rebalance); every variant is treated identically, so the
comparison between symbolizations is fair.

## Variants

| name | cutoffs | market-neutral | quantile scope |
|---|---|---|---|
| per_stock_20_40_60_80 | 20/40/60/80 | no | per-stock (baseline) |
| per_stock_15_35_65_85 | 15/35/65/85 | no | per-stock |
| per_stock_20_40_60_80_mnl | 20/40/60/80 | **yes** | per-stock |
| pooled_20_40_60_80 | 20/40/60/80 | no | **pooled (magnitude)** |
| pooled_15_35_65_85 | 15/35/65/85 | no | pooled |
| pooled_20_40_60_80_mnl | 20/40/60/80 | yes | pooled |
| correlation_ref | — | — | correlation distance |

## Report period (906 days, 31 rebalances)

| variant | Sharpe | ann_return | avg_k |
|---|---|---|---|
| per_stock_20_40_60_80 | 0.421 | 0.0460 | 2.81 |
| per_stock_15_35_65_85 | 0.632 | 0.0728 | 3.55 |
| **per_stock_20_40_60_80_mnl** | **0.869** | 0.1104 | 2.81 |
| pooled_20_40_60_80 | 0.669 | 0.0764 | 2.00 |
| pooled_15_35_65_85 | 0.599 | 0.0725 | 2.42 |
| pooled_20_40_60_80_mnl | 0.776 | 0.0965 | 2.26 |
| correlation_ref | 0.530 | 0.0615 | 3.00 |
| equal_weight | 0.746 | 0.0944 | — |

Sharpe diff vs equal_weight (95% CI): only the **baseline** is significantly worse
(-0.626, -0.061). Everything else includes 0; `per_stock_mnl` has the best point estimate
(+0.12).

## Tune period (2113 days, 88 rebalances) — a different regime

| variant | Sharpe | ann_return | avg_k |
|---|---|---|---|
| per_stock_20_40_60_80 | 0.863 | 0.1325 | 3.03 |
| per_stock_15_35_65_85 | 0.870 | 0.1323 | 3.12 |
| per_stock_20_40_60_80_mnl | 0.904 | 0.1432 | 2.89 |
| pooled_20_40_60_80 | 0.882 | 0.1370 | 2.52 |
| **pooled_15_35_65_85** | **0.981** | 0.1607 | 2.22 |
| pooled_20_40_60_80_mnl | 0.967 | 0.1564 | 2.66 |
| correlation_ref | 0.931 | 0.1435 | 3.02 |
| equal_weight | 0.946 | 0.1536 | — |

All CIs include 0. `pooled_mnl` (+0.02) and `pooled_15_35_65_85` (+0.035) have positive
point estimates; `per_stock` variants negative.

## What replicates, and what does not

**1. Preserving magnitude helps — consistently, but slightly.** Pooled symbolization beats
per-stock symbolization *in both periods* at the same cutoffs (report 0.669 vs 0.421; tune
0.882 vs 0.863). This supports the hypothesis that discarding magnitude cost signal. The
effect is small on the tune period and large on the report period, so "consistent
direction, unstable size".

**2. `pooled_mnl` is the only variant above equal weight in both periods** (0.776 vs 0.746;
0.967 vs 0.946) — but by a hair (+0.02 to +0.03 Sharpe) and with CIs that include 0.

**3. The report-period "winner" does not replicate — a selection artefact.** On the report
period `per_stock_mnl` looked best (0.869, +0.12 over equal weight). On the tune period it
is 4th (0.904, *below* equal weight 0.946). The top variant flips from `per_stock_mnl` to
`pooled_15_35_65_85`. Scanning six variants on 31 rebalances and reporting the best is
exactly the multiple-comparisons trap; **the ranking is not stable and should not be
trusted.** This is the headline caveat.

**4. The `EXPERIMENTS.md` conclusion is refined, not overturned.** There, correlation beat
alignment (0.506 vs 0.332) under the *baseline* symbolization. Here, with magnitude-aware
symbolization, alignment (0.669–0.776) beats correlation (0.530) on the report period. So
the earlier "alignment is the weak link" was really "**magnitude-discarding per-stock
symbolization is the weak link**".

**5. Equal weight remains hard to beat.** It is 0.746 / 0.946 across the two periods, and
no variant is *significantly* different from it in either.

## Caveats

- **Selection risk (the big one):** variants were chosen by looking at the same data they
  are scored on. Any "best variant" number is optimistic; the period-to-period flip proves
  it. A proper test needs pre-registration on a held-out period.
- The gate is bypassed, so this is not the exact production path (though the gate passes
  ~90% anyway).
- Fixed universe (survivorship), single asset class, monthly rebalancing, two periods only.
- Market-neutral + long-only: the book still earns the market; the symbolization is about
  *relative* patterns, so part of any gain is regime luck.

## Recommendation

Do **not** claim a winning symbolization. The defensible statements are: (a) magnitude-aware
symbolization is directionally better than per-stock and worth keeping as the default; (b)
no variant beats equal weight significantly; (c) the variant ranking is unstable, so pick on
a validation period, not this one. Suggested next: pre-register `pooled_mnl` vs `per_stock`
vs equal weight, then evaluate on a fresh window (or walk-forward with the choice frozen).
