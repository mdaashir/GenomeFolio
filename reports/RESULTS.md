# GenomeFolio — Results

Run: `uv run python scripts/run_full_report.py` (1000 gate draws, 10 bps base cost).
Raw tables: `reports/metrics.csv`, `reports/metrics_gate_passed_only.csv`,
`reports/bootstrap_ci.csv`, `reports/cost_sweep.csv`, `reports/band_sweep_ari.csv`,
`reports/turnover.csv`. Machine log: `reports/run.log`.

## Setup

- Universe: 20 fixed cross-sector US large-caps, 3020 trading days (2013 → 2024).
- Split: tune 2113 days / **report 906 days** (May 2022 → Nov 2024), **~31 monthly
  rebalances**.
- Symbolization: per-stock training-window 20/40/60/80 quantiles → 5 symbols.
- Alignment: banded Needleman-Wunsch, graded matrix, affine gaps, band w = 5.
- Clustering: hierarchical average linkage, k by silhouette over [2, min(6, ⌊N/3⌋)],
  min cluster size 3.
- Null gate: circular-shift, real silhouette vs 95th percentile of the null.
- Portfolio: two-stage (equal across clusters, equal within), per-stock cap 10%.
- Costs: turnover = Σ|Δw|, cost = 10 bps × turnover.

## Headline

**The alignment portfolio did not beat the baselines — it underperformed most of them,
and the gap is statistically significant against several.** Clusters were unstable across
alignment bands, and the null gate almost always passed. This is a negative result for
this method on this universe, on this period.

## Metrics (all 31 rebalances, net 10 bps)

| method | ann_return | ann_vol | sharpe | max_dd | avg_turnover | gate_pass |
|---|---|---|---|---|---|---|
| **alignment** | 0.036 | 0.133 | **0.332** | -0.179 | 0.299 | 0.903 |
| equal_weight | 0.094 | 0.133 | 0.746 | -0.157 | 0.032 | — |
| min_variance | 0.063 | 0.112 | 0.599 | -0.126 | 0.131 | — |
| mean_variance | 0.110 | 0.122 | **0.919** | -0.122 | 0.258 | — |
| correlation | 0.061 | 0.135 | 0.506 | -0.159 | 0.170 | — |
| random (same sizes) | 0.093 | 0.134 | 0.730 | -0.154 | 0.451 | — |
| sector | 0.094 | 0.129 | 0.758 | -0.156 | 0.032 | — |

Alignment has the **lowest Sharpe of all seven methods**, even before costs (cost sweep
below shows Sharpe 0.359 at 0 bps). Its turnover (0.30) is ~9× equal-weight's (0.03).

## Paired block-bootstrap 95% CIs (alignment − baseline)

| baseline | Sharpe diff | Return diff |
|---|---|---|
| equal_weight | **(-0.688, -0.193)** | **(-0.099, -0.028)** |
| mean_variance | **(-1.160, -0.165)** | **(-0.154, -0.016)** |
| random | **(-0.698, -0.094)** | **(-0.104, -0.014)** |
| sector | **(-0.682, -0.239)** | **(-0.095, -0.031)** |
| min_variance | (-0.847, 0.226) | (-0.103, 0.039) |
| correlation | (-0.545, 0.159) | (-0.084, 0.020) |

**CIs exclude 0 (alignment is worse) against equal-weight, mean-variance, random
same-size clusters, and sector clusters.** It is indistinguishable from min-variance and
correlation. The **random-cluster control is the important one**: random groupings of the
*same sizes* through the *same two-stage allocation* beat the alignment groupings, so it
is the alignment itself — not the size structure or the allocator — that hurts.

## Descriptive view: gate-passed rebalances only (no CI)

Baselines restricted to the same rebalances (descriptive only; non-contiguous months break
the bootstrap block structure).

| method | ann_return | ann_vol | sharpe | max_dd |
|---|---|---|---|---|
| alignment | 0.052 | 0.137 | 0.438 | -0.179 |
| equal_weight | 0.118 | 0.136 | 0.884 | -0.157 |
| mean_variance | 0.151 | 0.124 | 1.195 | -0.122 |
| random | 0.111 | 0.137 | 0.836 | -0.154 |
| sector | 0.112 | 0.133 | 0.866 | -0.156 |

Restricting to gate-passed months does not rescue the method; it still trails every
comparison.

## Null gate: passes 90% of the time

Gate pass fraction = **0.903** (28 of 31 rebalances). At the 95th percentile, chance alone
would give ~5%. This is the expected market-factor artefact: equities move together, so
real cross-stock similarity beats a circular-shift null that removes only cross-alignment.
Consequence: **the gate provides almost no protection here** — it is signalling "these
stocks co-move", which is trivially true, not "these clusters are meaningful".

## Band sensitivity: the clusters are not stable

| w | k | silhouette |
|---|---|---|
| 2 | 2 | 0.044 |
| 5 | 5 | 0.034 |
| 10 | 2 | 0.024 |

Adjusted Rand Index between band solutions:

| | w=2 | w=5 | w=10 |
|---|---|---|---|
| w=2 | 1.00 | 0.25 | 0.40 |
| w=5 | 0.25 | 1.00 | 0.07 |
| w=10 | 0.40 | 0.07 | 1.00 |

The number of clusters changes with the band and the partitions barely agree (ARI 0.07–0.40).
The solution is an artefact of the band, which undermines any interpretation of the
clusters. Silhouettes are also very low (0.02–0.04) in absolute terms.

## Cost sensitivity (alignment)

| bps | ann_return | sharpe |
|---|---|---|
| 0 | 0.0397 | 0.359 |
| 10 | 0.0359 | 0.332 |
| 25 | 0.0304 | 0.291 |

Costs matter but are not the main story — alignment is already the worst performer at
0 bps.

## Why it likely underperforms (hypotheses, not conclusions)

1. **Concentration via cluster size.** Equal weight *across clusters* means a 3-stock
   cluster each gets ~6.7% while a 7-stock cluster gets ~2.9% — small clusters are
   overweighted. Risk-parity across clusters or size-aware weighting would test this.
2. **Per-stock quantile normalisation discards magnitude**, so the strings may encode
   *timing of moves* that is largely noise, not a persistent, tradable similarity.
3. **Unstable clusters** (low cross-band ARI) churn membership month to month → high
   turnover and no stable signal.
4. **The gate doesn't filter**, so clusters are used almost every month regardless of
   whether they mean anything.

## Caveats (read before quoting any number)

- **A backtest is not proof.** One path through history on a **fixed universe** → the
  survivors are the ones tested (survivorship bias).
- **The post-split report period is short** (~2.5 years, 31 rebalances), so the CIs are
  wide — but wide CIs still exclude 0 against four baselines.
- **This is a single period** (2022–2024, largely a specific market regime).
- Forecast tilt was **not** the cause: default `cluster_method` is `equal`, so the ARIMA
  forecast was not driving weights in this run.

## Suggested next experiments

1. Sweep the portfolio layer: `risk_parity` across clusters, `inverse_vol` within,
   and size-aware cluster weights — does the concentration hypothesis hold?
2. Compare alignment distance against a plain correlation-distance with the *same*
   constrained-k clustering, to isolate the alignment step.
3. Re-run per-stock quantile thresholds and the `market_neutral` ablation to see whether
   magnitude matters.
4. A longer report period (or rolling multi-window evaluation) to shrink the CIs.
