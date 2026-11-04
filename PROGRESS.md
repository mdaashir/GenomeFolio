# GenomeFolio — Progress

Sequence-alignment clustering of stock return patterns, followed by a per-cluster forecast.

**Pipeline:** data → symbolize → align → cluster → **forecast** (stops here).

## Pipeline status

| Stage | Module | Status |
|-------|--------|--------|
| 1. Data (prices → returns) | `src/data.py` | done, tested |
| 2. Symbolization (5-symbol alphabet) | `src/symbolize.py` | done, tested |
| 3. Alignment (banded NW, affine gaps) | `src/align.py` | done, tested |
| 4. Clustering (constrained k, band sweep) | `src/cluster.py` | done, tested |
| 5. Forecast (per-cluster ARIMA/spline) | `src/forecast.py` | done, tested |
| — Orchestration / CLI | `src/pipeline.py`, `src/cli.py` | done, tested |

Test suite: `uv run pytest`. Run: `uv run genomefolio --use-fixture`.

Documentation: `docs/` contains abstract, explanation, demo, architecture and configuration
guides, each in Markdown with a matching PDF in `docs/pdf/`.

UI: `app.py` is a Streamlit app that runs the pipeline and shows every phase (data,
symbolization, raw scores, pairwise distance, clustering, forecast) with graphs and tables.
Run it with `uv run streamlit run app.py`.

## Removed in the "present" simplification

The circular-shift null gate, the two-stage portfolio, the rolling update, the walk-forward
backtest, transaction-cost and evaluation helpers, all experiments/reports that depended on
them, and `config/sectors.yaml`. The pipeline now goes **straight from clustering to the
forecast** — no gate decision, no allocation, no backtest.

## Parameters chosen and why

All in `config/params.yaml`; nothing hardcoded.

- **Universe**: 20 fixed cross-sector US large-caps, 2013–2018 history. Fixed universe →
  **survivorship bias is documented, not hidden**.
- **Symbolization**: per-stock **20/40/60/80** training-window quantiles → 5 symbols.
  Per-stock normalisation strips each stock's own volatility, so strings encode *when*
  moves happen (shape/timing), not how big — deliberate, since the goal is pattern
  similarity, not correlation.
- **Alignment**: graded matrix (`+2` match, else `−|ord(s) − ord(t)|`), **affine gaps**
  (open −1.0 / extend −0.4), **band w = 5** (lets a pattern shift a few days, hard-stops
  unrelated pairings). `band_sweep = [2, 5, 10]` for stability checks.
- **Distance**: raw score normalised by the two self-scores (`1 − s/√(self_i·self_j)`).
  **Not claimed to be a metric** — asserted symmetric and non-negative only.
- **Clustering**: hierarchical **average** linkage; k by silhouette over
  `[2, min(6, floor(N/3))]` with **min cluster size 3**.
- **Forecast**: per-cluster ARIMA on the equal-weight composite, forecasting the
  **~21-day cumulative** return; order by AIC over `p, q ≤ 2`, `d` from an ADF test on the
  training window; falls back to the historical mean if no fit converges.

## Design decisions (for the report)

- **Why sequence alignment, not correlation**: correlation only sees co-movement over the
  whole window; alignment can match a pattern that occurs a few days earlier or later.
- **Why per-stock quantiles**: makes strings comparable across stocks of different
  volatility; discards magnitude on purpose.
- **Why a graded matrix**: near-misses (big-up vs small-up) should cost less than
  opposites (big-up vs big-down).
- **Why stop at the forecast**: the forecast horizon is explicit and testable; turning it
  into portfolios/backtests is out of scope for this version.

## Known issues / caveats

- **Forecasts are not proof.** Daily returns are weakly predictable; ARIMA is close to the
  mean, so report every forecast with its horizon and treat it as a modest signal.
- **Window choice matters**: symbols and thresholds use only the trailing training window;
  a different window yields different clusters.
- **Cluster count is band-sensitive**: check the band sweep / silhouette before trusting a
  single configuration.
- **Survivorship bias**: the fixed universe contains names that survived to today.
- **No look-ahead**: every stage at time *t* uses only data up to *t*.

## Next steps

1. Evaluate forecast accuracy out of sample (hold out the last ~30% via `split`).
2. Sweep symbolization cutoffs and the band width and report cluster stability.
3. Consider richer symbolization (rolling z-score) if patterns look regime-dependent.
