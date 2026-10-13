# GenomeFolio

Sequence-alignment clustering of stock return patterns → a forecast-tilted, diversified
walk-forward portfolio with a bias-aware backtest.

## Install

```bash
uv sync
```

## Quickstart

```bash
uv run genomefolio --use-fixture --band-only      # offline smoke (band-sweep stability)
uv run genomefolio                                # full run (downloads the 20-stock universe)
uv run pytest                                     # test suite
```

## What it does

1. **Symbolise** each stock's daily returns over the training window into a 5-symbol
   alphabet (big-down … big-up).
2. **Align** pairs of symbol strings with banded Needleman-Wunsch (graded matrix, affine
   gaps) and turn scores into a normalised distance.
3. **Cluster** with hierarchical average linkage under a constrained-k rule.
4. **Gate** clustering against a circular-shift null (and a factor-model variant).
5. **Allocate** two-stage (across clusters, then within), cap-constrained.
6. **Backtest** walk-forward with transaction costs and paired block-bootstrap CIs.

See [Method](method.md) for the details, [Configuration](config.md) for every knob, and
[Results](results.md) for the study outcomes.
