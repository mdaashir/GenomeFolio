# GenomeFolio — Package Abstract

## Introduction

Financial return series are usually grouped with correlation. Correlation measures how much
two series move together over a whole window, but it cannot recognise the same *pattern*
when it appears a few days earlier in one stock and a few days later in another. This
project instead treats each stock's return history as a **symbolic sequence**, groups stocks
by **sequence similarity**, and forecasts each group's behaviour.

## Description

GenomeFolio implements a five-stage pipeline.

1. **Data** — daily adjusted prices for a fixed universe of US large-cap stocks are converted
   to simple daily returns on their common trading days.
2. **Symbolization** — each stock's returns over the trailing training window become a
   five-symbol alphabet: big-down, small-down, flat, small-up, big-up. Thresholds are
   per-stock quantiles of the training window, so one stock's volatility does not dominate
   the comparison.
3. **Alignment** — every pair of symbol strings is aligned with a banded **Needleman-Wunsch**
   global alignment using affine gaps and a graded substitution matrix. The band limits how
   far a pattern may shift, so unrelated days cannot be paired.
4. **Clustering** — the normalised alignment scores form a distance matrix, which is
   clustered with hierarchical average linkage. The number of clusters is chosen by
   silhouette under a constrained search with a minimum cluster size.
5. **Forecast** — each cluster's equal-weight composite return is forecast over the coming
   21 trading days with ARIMA; a spline model is available as an alternative.

The pipeline stops at the forecast. It deliberately contains no null-hypothesis gate, no
portfolio construction and no backtest.

## Datasets Used

- **Primary** — daily adjusted close downloaded with **yfinance** from **2013 to 2018**
  (one request, cached locally), covering 20 stocks: AAPL, MSFT, INTC, CSCO, JPM, VZ, T,
  XOM, CVX, CAT, JNJ, PFE, MRK, PG, KO, WMT, DIS, HD, MCD, NKE.
- **Offline fixture** — a committed 10-stock sample covering five years, used by the tests
  and by the offline demo so the project runs without network access.

## Expected Outcome

The expected outcome is (a) interpretable groups of stocks whose return patterns are
structurally similar and (b) a 21-day forecast for each group's composite. Because daily
returns are only weakly predictable, the forecast is reported as a modest signal with its
horizon, and limitations — survivorship, window sensitivity and weak predictability — are
documented rather than hidden.

