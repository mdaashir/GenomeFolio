# GenomeFolio — Pre-registered Holdout Evaluation (result)

Spec: `reports/PREREGISTRATION.md` (written before this ran). Run:
`uv run python scripts/run_holdout_eval.py`. Tables: `reports/holdout_eval.csv`,
`reports/holdout_ci.csv`, log `reports/holdout.log`.

Held-out universe (disjoint from the original 20):
`NVDA AMD QCOM TXN ORCL ADBE CRM IBM ABT BMY LLY AMGN GILD TGT LOW SBUX BA GE MMM HON`,
2013–2024, **131 rebalances**. One frozen confirmatory comparison, no tuning.

## Result

| arm | Sharpe | ann_return | ann_vol | max_dd |
|---|---|---|---|---|
| per_stock (context) | 1.065 | 0.1909 | 0.179 | -0.295 |
| **pooled_mnl** | 1.059 | 0.2056 | 0.195 | -0.305 |
| equal_weight | 1.039 | 0.1969 | 0.191 | -0.307 |

Paired block-bootstrap 95% CI vs equal_weight (5,000 resamples):

| arm | Sharpe diff | Return diff |
|---|---|---|
| **pooled_mnl** | **(-0.064, 0.107)** | (-0.011, 0.031) |
| per_stock | (-0.107, 0.163) | (-0.036, 0.023) |

## Verdict: H1 NOT SUPPORTED

Per the pre-registered decision rule, H1 required the 95% CI on the Sharpe difference
`pooled_mnl − equal_weight` to **exclude 0 and be positive**. It is `(-0.064, 0.107)` —
it **includes 0**. **H1 is not supported.**

## What this means

- **No symbolization reliably beats equal weight.** On the held-out universe the
  magnitude-aware, market-neutral arm lands at +0.02 Sharpe over equal weight — the same
  sign but far from significant, and *not* the +0.12 seen on the original report period.
  The report-period "win" (`per_stock_mnl` at 0.869) was selection, as the tune-period
  check suspected.
- **The magnitude hypothesis is not confirmed either.** Here `per_stock` (1.065) and
  `pooled_mnl` (1.059) are statistically tied. Whatever small edge magnitude-aware
  symbolization showed on the two original periods does not reappear on a fresh universe.
- **Return vs risk:** `pooled_mnl` has the best *annualized return* (0.206 vs 0.197) but
  the highest volatility (0.195 vs 0.191), so it gives up the Sharpe edge — consistent with
  the tilt toward more concentrated positions rather than a genuine risk-adjusted gain.

This is the clean, honest negative result the whole pipeline was built to produce: on
held-out data, a sequence-alignment clustering portfolio is **indistinguishable from
equal weight**, and nothing about the symbolization choice changes that materially.

## Caveats

- Macro period overlaps the original study (2013–2024); only the tickers are new. A truly
  disjoint time period was not available in this dataset.
- Survivorship: a fixed universe of 20 survivors.
- Null gate bypassed (frozen in the pre-registration); it passes ~90% anyway.
- One universe, one shot: it is evidence, not proof. But it is the one comparison that was
  decided in advance, so it carries more weight than the exploratory numbers.

## Bottom line for the report

Present the study as a **methodological pipeline with a null result**: the machinery is
correct, reproducible, and bias-aware; the method does not beat equal weight on this
evidence; the exploratory gains did not survive a pre-registered holdout. The value is in
the rigour (bias controls, replication checks, pre-registration), not in a performance
claim.
