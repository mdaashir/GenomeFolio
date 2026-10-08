# GenomeFolio — Time-disjoint Out-of-Sample Result

Spec: `reports/PREREGISTRATION_TIME.md` (written before fetching). Run:
`uv run python scripts/run_time_oos.py`. Tables: `reports/time_oos_eval.csv`,
`reports/time_oos_ci.csv`, log `reports/time_oos.log`.

Original 20-stock universe, **2005-2012** (unseen era, includes the 2008 crisis), **83
rebalances**, frozen spec, one shot.

## Result

| arm | Sharpe | ann_return | ann_vol |
|---|---|---|---|
| per_stock (context) | 0.635 | 0.111 | 0.196 |
| **pooled_mnl** | **0.707** | 0.135 | 0.211 |
| equal_weight | 0.639 | 0.119 | 0.211 |

`pooled_mnl − equal_weight` Sharpe diff 95% CI: **(-0.006, 0.141)** — includes 0.

## Verdict: H2 NOT SUPPORTED

CI includes 0 (narrowly). It is the closest to significance of any confirmatory test, but
by the pre-registered rule it does not pass.

## The pattern across all four evaluations

`pooled_mnl` (magnitude-aware + market-neutral) has a **higher Sharpe than equal weight in
every evaluation**, but **never significantly**:

| evaluation | period | rebalances | pooled_mnl | equal_weight | diff | CI includes 0? |
|---|---|---|---|---|---|---|
| report (in-sample) | 2022–2024 | 31 | 0.776 | 0.746 | +0.030 | yes |
| tune (in-sample) | 2013–2021 | 88 | 0.967 | 0.946 | +0.021 | yes |
| holdout universe | 2013–2024 | 131 | 1.059 | 1.039 | +0.020 | yes |
| time OOS | 2005–2012 | 83 | 0.707 | 0.639 | +0.068 | yes |

Three of the four are genuinely out-of-sample for the *choice*; the sign is consistent,
the magnitude is small, and every CI spans 0.

## Interpretation

- **Not confirmed, but not refuted.** Four independent-ish evaluations all point the same
  way by a small margin. That is more consistent than pure noise, but with 31–131
  rebalances per test the power is too low to call it. Do **not** pool these into a single
  p-value: the evaluations overlap in universe and time, so they are not independent.
- **The earlier strong in-sample claim is gone.** `per_stock_mnl`'s 0.869 on the report
  period remains an outlier that did not replicate; `pooled_mnl` is the arm with the
  consistent small edge.
- **Risk trade-off persists:** `pooled_mnl` carries the highest volatility in every run, so
  the return gains are not risk-free.

## Honest conclusion

The pre-registered tests did not confirm a beat over equal weight. The defensible statement
is: *magnitude-aware, market-neutral symbolization shows a small, directionally consistent
risk-adjusted improvement over equal weight across four evaluations, but it is not
statistically significant in any of them and the effect size is within the noise of a
~2–8 year sample.* Claiming a win would be over-reach; claiming the method is worthless
would also be over-reach. The pipeline is sound and the question is simply underpowered.

## Caveats

- Overlapping universes/periods across evaluations → not independent; no meta-analysis.
- Survivorship (fixed universes), single asset class, monthly rebalancing.
- Null gate bypassed in the OOS tests (frozen); it passes ~90% anyway.
- One shot per test, as pre-registered.
