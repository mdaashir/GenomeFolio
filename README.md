# GenomeFolio

Sequence-alignment clustering of stock return patterns, followed by a per-cluster forecast.

Stocks' daily returns are symbolised into a 5-symbol alphabet over a training window, aligned
pairwise with **banded global alignment** (Needleman-Wunsch, affine gaps, graded substitution
matrix), turned into a normalised distance, and clustered with **hierarchical average linkage**
under a **constrained k** rule. Each cluster's equal-weight composite return is then
**forecast** over the coming horizon (ARIMA or spline).

The pipeline stops at the forecast: there is **no circular-shift null gate** and **no
portfolio / walk-forward backtest** downstream.

## Setup

```
uv sync
```

## Run

```
# Full dataset (downloads/caches prices on first run)
uv run genomefolio

# Offline run on the committed 10-stock fixture
uv run genomefolio --use-fixture

# Write clusters + forecasts to JSON
uv run genomefolio --use-fixture --out forecasts.json
```

## Tests

```
uv run pytest
```

## Layout

```
config/    params.yaml (all knobs; nothing hardcoded)
src/       config, data, symbolize, align, cluster, forecast, pipeline, cli
tests/     pytest suite per module
notebooks/ exploration only
data/      gitignored raw/cache; data/sample/ is the committed offline fixture
```
