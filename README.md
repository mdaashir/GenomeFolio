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

## Interactive UI (Streamlit)

The app shows the output of every pipeline phase, each with a graph and a table: data
(cumulative returns), symbolization (symbol heatmap and frequency), alignment (raw score
matrix), **pairwise distance matrix** after the score→distance conversion, clustering
(sizes, silhouette curve, dendrogram), forecast paths and the band-stability check.

```
uv sync --group ui
uv run --group ui streamlit run app.py
```

## Tests

```
uv run pytest
```

## Documentation

Full documentation lives in `docs/`, in Markdown with matching PDFs in `docs/pdf/`
(rendered with the Mathematical Modelling Lab abstract template):

- `docs/abstract.md` — `docs/pdf/abstract.pdf`
- `docs/explanation.md` — `docs/pdf/explanation.pdf`
- `docs/demo.md` — `docs/pdf/demo.pdf`
- `docs/architecture.md` — `docs/pdf/architecture.pdf`
- `docs/configuration.md` — `docs/pdf/configuration.pdf`

## Layout

```
config/    params.yaml (all knobs; nothing hardcoded)
src/       config, data, symbolize, align, cluster, forecast, pipeline, plots, cli
app.py     Streamlit UI (every phase, graphs + tables)
tests/     pytest suite per module
notebooks/ exploration only
data/      gitignored raw/cache; data/sample/ is the committed offline fixture
```
