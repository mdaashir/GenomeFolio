# GenomeFolio

Sequence-alignment clustering of stock return patterns, with a forecast-tilted,
diversified walk-forward portfolio and a bias-aware backtest.

Stocks' daily returns are symbolised into a 5-symbol alphabet, aligned pairwise with
**banded global alignment** (Needleman-Wunsch, affine gaps, graded substitution matrix),
turned into a normalised distance, and clustered with **hierarchical average linkage**
under a **constrained k** rule. A **circular-shift null gate** checks that clustering
beats chance. Clusters feed a **two-stage diversified portfolio** (allocate across
clusters, then within them under per-stock caps), lightly tilted by a per-cluster ARIMA
forecast, rebalanced monthly in a walk-forward backtest that charges turnover and
reports **paired block-bootstrap** confidence intervals against several baselines.

See `AGENTS.md` for the project brief and `PROGRESS.md` for decisions, caveats and
results.

## Setup

```
uv sync
```

## Run

```
# Full dataset (downloads/caches prices on first run)
uv run python -m src.backtest

# Offline smoke run on the committed 10-stock fixture
uv run python -m src.backtest --use-fixture --draws 30

# Only the band-sweep stability report
uv run python -m src.backtest --use-fixture --band-only
```

## Tests

```
uv run pytest
```

## Layout

```
config/    params.yaml, sectors.yaml (all knobs; nothing hardcoded)
src/       data, symbolize, align, cluster, null_test, forecast, portfolio, rolling, backtest
tests/     pytest suite per module
notebooks/ exploration only
data/      gitignored raw/cache; data/sample/ is the committed offline fixture
```
