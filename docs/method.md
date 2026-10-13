# Method

## Pipeline

| Stage | Module | Notes |
|---|---|---|
| Data | `genomefolio/data.py` | adjusted close, daily returns on common days, chronological split |
| Symbolise | `genomefolio/symbolize.py` | per-stock / pooled quantiles, or rolling z-score |
| Align | `genomefolio/align.py` | banded Needleman-Wunsch, affine gaps, graded matrix |
| Cluster | `genomefolio/cluster.py` | average linkage, constrained k, band sweep, consensus |
| Gate | `genomefolio/null_test.py` | circular-shift null, factor-model variant |
| Forecast | `genomefolio/forecast.py` | per-cluster ARIMA over the rebalance horizon |
| Portfolio | `genomefolio/portfolio.py` | two-stage allocation, caps, baselines |
| Backtest | `genomefolio/backtest.py` | walk-forward, costs, paired bootstrap |
| Evaluation | `genomefolio/evaluation.py` | deflated Sharpe, stationary bootstrap |

## Alignment

Symbols are ordinals `0..4`. The substitution matrix is graded and symmetric:

```
score(s, t) = match_score         if s == t
            = -|s - t|            otherwise
```

Global alignment uses affine gaps and a band `|i - j| <= w`. The distance is normalised by
the two self-scores:

```
similarity(a, b) = score(a, b) / sqrt(self(a) * self(b))
distance(a, b)   = 1 - similarity(a, b)
```

This distance is **not** claimed to be a metric — it is only asserted symmetric and
non-negative, which is all average linkage needs.

## Clustering

Average linkage on the distance matrix; `k` chosen by silhouette over
`[k_min, min(k_max, floor(N / divisor))]` subject to every cluster having at least
`min_cluster_size` members. `consensus_distance` clusters on the **median** distance across
band widths to tame band sensitivity.

## Null gate

The real silhouette must beat the 95th percentile of **circular-shift** null silhouettes
(each string shifted by a random offset larger than the band), with the null run through
the same constrained-k selection. Because equities share a market factor, the raw gate
passes ~90% of the time; `factor_neutral_symbols` gates on factor-residual returns instead.

## Evaluation

- **Deflated Sharpe** (Bailey & López de Prado): shrinks a Sharpe that came out of a large
  specification search.
- **Paired block bootstrap** (fixed or Politis-White block length) and a **stationary
  bootstrap** alternative, applied to both method and baseline on identical resamples.

## No look-ahead

Every stage at time *t* uses only data up to *t*. Quantile thresholds, fits, the gate and
the cluster labels are all computed on the trailing training window; the tune/report split
is chronological.
