# GenomeFolio — Configuration Reference

Every parameter lives in `config/params.yaml`. The loader (`src/config.py`) validates the
values at startup, so a bad configuration fails immediately. Defaults are shown in
parentheses.

## seed (42)

Master random seed. All randomness — the bootstrap-era code is gone, but the seed is kept
for reproducibility of any future stochastic step.

## universe

- `tickers` — the fixed stock universe (20 cross-sector US large-caps by default).
- `start`, `end` — ISO dates for the download range (`2013-01-01`, `2025-01-01`).
- `cache_dir` — where downloaded prices are cached (`data/cache`). Gitignored.
- `fixture_path` — the committed offline sample (`data/sample/prices.csv`).
- `min_history_years` — a documented expectation for the history length (10).

## split

- `enabled` (`true`) — whether a chronological split is defined.
- `tune_fraction` (`0.70`) — fraction of history used first. The split helper is available
  for holding out a forecast-evaluation period; the default pipeline uses only the trailing
  training window.

## symbolize

- `cutoffs` (`[0.20, 0.40, 0.60, 0.80]`) — per-stock quantiles that cut returns into the five
  symbols. Must be four strictly increasing values inside (0, 1).
- `cutoffs_ablation` (`[0.15, 0.35, 0.65, 0.85]`) — an alternative band used for sensitivity
  experiments.
- `market_neutral` (`false`) — subtract the equal-weight universe return before symbolizing.
- `quantile_scope` (`per_stock`) — `per_stock` normalises each stock's own volatility away;
  `pooled` computes thresholds from all stocks pooled and keeps relative volatility.

## align

- `match_score` (`2.0`) — score for a matching symbol. Distinct symbols score
  `-|s - t|`, so adjacent symbols cost less than opposites.
- `gap_open` (`-1.0`) — cost of opening a gap.
- `gap_extend` (`-0.4`) — cost of each additional gap position (affine gaps).
- `band` (`5`) — maximum `|i - j|` in the dynamic program; caps how far a pattern may shift.
- `band_sweep` (`[2, 5, 10]`) — band widths used by the stability check.
- `use_numba` (`true`) — compile the dynamic program; set false for the pure-Python path.

## cluster

- `linkage` (`average`) — SciPy linkage method. Average linkage is the default because it
  works directly on a distance matrix without Euclidean assumptions.
- `k_min` (`2`) — smallest number of clusters considered.
- `k_max` (`6`) — largest number of clusters considered.
- `k_cap_divisor` (`3`) — the effective cap is `min(k_max, floor(N / divisor))`, so small
  universes cannot be over-split.
- `min_cluster_size` (`3`) — every cluster must contain at least this many stocks for a
  candidate to be accepted.

## window

- `train_days` (`252`) — length of the trailing training window, in trading days.
- `forecast_horizon` (`21`) — number of trading days ahead that the forecast covers (about
  one month).

## forecast

- `model` (`arima`) — `arima` or `spline`.
- `max_p` (`2`), `max_q` (`2`) — bounds of the AIC grid for the AR and MA orders.
- `seasonal` (`false`) — reserved for a seasonal ARIMA variant.
- `fallback` (`historical_mean`) — used when no model converges or the series is degenerate.
