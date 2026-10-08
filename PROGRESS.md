# GenomeFolio — Progress

Sequence-alignment clustering of financial return patterns → diversified, walk-forward
portfolio with a bias-aware backtest.

## Pipeline status

| Stage | Module | Status |
|-------|--------|--------|
| 1. Data (prices → returns) | `src/data.py` | done, tested |
| 2. Symbolization (5-symbol alphabet) | `src/symbolize.py` | done, tested |
| 3. Alignment (banded NW, affine gaps) | `src/align.py` | done, tested |
| 4. Clustering (constrained k, band sweep) | `src/cluster.py` | done, tested |
| 5. Shuffled-null gate (circular shift) | `src/null_test.py` | done, tested |
| 6. Forecast (per-cluster ARIMA) | `src/forecast.py` | done, tested |
| 7. Portfolio (two-stage + caps) | `src/portfolio.py` | done, tested |
| 8. Rolling update | `src/rolling.py` | done |
| 9. Backtest + metrics + bootstrap | `src/backtest.py` | done, tested |
| — Evaluation / write-up | `reports/RESULTS.md`, this file | done |

Test suite: `uv run pytest` (60 tests). Smoke run:
`uv run python -m src.backtest --use-fixture --draws 30`.

## Parameters chosen and why

All in `config/params.yaml`; nothing hardcoded.

- **Universe**: 20 fixed cross-sector US large-caps, 10+ year history. Fixed universe →
  **survivorship bias is documented, not hidden** (the names that survived to today are
  the ones we test).
- **Symbolization**: per-stock **20/40/60/80** training-window quantiles → 5 symbols.
  Per-stock normalisation strips each stock's own volatility, so strings encode *when*
  moves happen (shape/timing), not how big — deliberate, since the goal is pattern
  similarity, not correlation. Ablations: **15/35/65/85** cutoffs and **market-neutral**
  returns (stock − equal-weight universe).
- **Alignment**: graded matrix (`+2` match, else `−|ord(s) − ord(t)|`), **affine gaps**
  (open −1.0 / extend −0.4), **band w = 5**. The band lets a pattern shift a few days but
  hard-stops pairing unrelated days. Sweep **w ∈ {2, 5, 10}** for stability.
- **Distance**: raw score normalised by the two self-scores (`1 − s/√(self_i·self_j)`).
  **Not claimed to be a metric** — asserted symmetric and non-negative only; average
  linkage needs no metric.
- **Clustering**: hierarchical **average** linkage; k by silhouette over
  `[2, min(6, floor(N/3))]` with **min cluster size 3** (silhouette favours k = 2 with
  ~20 stocks otherwise).
- **Null gate**: real silhouette must beat the **95th percentile** of **circular-shift**
  null silhouettes, with the null run through the **same constrained k selection**.
  Circular shifts (> band) preserve each string's autocorrelation and volatility
  clustering and break only cross-stock alignment — an i.i.d. shuffle destroys both and
  makes the null too easy to beat. Per-pair **BH-FDR is a diagnostic only**.
- **Forecast**: per-cluster ARIMA forecasting the **~21-day cumulative** return (matches
  the monthly rebalance), order by AIC over `p, q ≤ 2`, `d` from an ADF test on the
  training window; falls back to the historical mean if no fit converges. Treated as a
  **modest tilt only** — daily returns are weakly predictable.
- **Portfolio**: two-stage — across clusters (equal / risk-parity / light forecast tilt),
  then within cluster (equal / inverse-vol), under **cap = max(10%, 1.5/N)**.
- **Costs**: **turnover = Σ|Δw|** (total traded weight, buys plus sells); cost =
  **10 bps × turnover**; sweep **0 → 25 bps**.
- **Bootstrap**: **paired** block bootstrap (same resampled blocks for method and
  baseline), block ≈ 20 days (or Politis-White), 5,000 resamples, 95% CIs.
- **No data-snooping**: tune on the first ~70% of history, report on the untouched last
  ~30% (full dataset only; the fixture runs with the split disabled).

## Design decisions (for the report)

- **Why sequence alignment, not correlation**: correlation only sees co-movement over the
  whole window; alignment can match a pattern that occurs a few days earlier or later.
- **Why per-stock quantiles**: makes strings comparable across stocks of different
  volatility; discards magnitude on purpose.
- **Why a graded matrix**: near-misses (big-up vs small-up) should cost less than
  opposites (big-up vs big-down).
- **Why two-stage allocation**: with 15–25 stocks a single global optimiser mostly
  amplifies covariance-estimation noise; two-stage keeps "mix across clusters" central.
- **Why the null gate**: guards against clustering pure noise; the pass fraction is read
  against the ~5% chance rate.

## Known issues / caveats

- **Backtest ≠ proof.** Results are one path through history on a fixed universe; flag
  overfitting and look-ahead wherever relevant.
- **Post-split period is short** (~30% of 10+ years ≈ 2.5–3 years, ~30 rebalances), so
  the block-bootstrap CIs are **wide**.
- **Gate pass fraction runs far above 5%**: 0.90 on the full run (and ~0.49 on the
  fixture). Equities share a market factor, so real cross-stock similarity beats a null
  that removes only cross-alignment. It means "similarity exists", not "clusters add
  value" — the gate provides almost no protection here.
- **Alignment O(n²) per pair**: numba-accelerated; the DP still allocates full
  `(n+1)²` matrices, so keep windows at the configured 252 days. The null gate cost is
  controlled by the draw count (~200 tuning / ~1000 reported).
- **ARIMA on daily returns is close to the mean**; the forecast is a small tilt, not a
  signal.

## Results (full run)

20 stocks, 3020 days, report period 906 days (~31 rebalances), 1000 gate draws, 10 bps.
Full tables and interpretation in `reports/RESULTS.md`.

**Negative result.** The alignment portfolio had the **lowest Sharpe of all seven methods**
(0.33 vs equal-weight 0.75, mean-variance 0.92), and the paired block bootstrap shows it is
**significantly worse** than equal-weight, mean-variance, random same-size clusters and
sector clusters (CIs exclude 0). Random groupings of the *same sizes* through the *same
two-stage allocation* beat the alignment groupings, so it is the alignment itself that
hurts. Clusters were unstable across bands (ARI 0.07–0.40) and the gate passed 90% of the
time.

## Next steps

1. Test the concentration hypothesis: `risk_parity` across clusters, `inverse_vol` within,
   and size-aware cluster weights.
2. Isolate the alignment step: same constrained-k clustering on plain correlation distance.
3. Re-run the quantile-threshold and `market_neutral` ablations.
4. Optional: replace the crude Politis-White rule with the full automatic selection.
