# GenomeFolio — Consolidated Report

Sequence-alignment clustering of financial return patterns → diversified walk-forward
portfolio. Full study, all results, all caveats.

Supporting documents: `PROGRESS.md` (decision log), `RESULTS.md` (main run),
`EXPERIMENTS.md` (portfolio sweep + isolation), `ABLATIONS.md` (symbolization),
`PREREGISTRATION.md` + `HOLDOUT.md` (ticker-disjoint), `PREREGISTRATION_TIME.md` +
`TIME_OOS.md` (time-disjoint).

---

## 1. Question

Group 15–25 stocks by the **similarity of their return patterns** — not correlation —
using bioinformatics-style sequence alignment, build a portfolio that mixes stocks across
clusters, rebalance as new data arrives, and test whether it beats traditional portfolio
methods on return-for-risk.

## 2. Method

| Stage | Choice |
|---|---|
| Data | Fixed cross-sector US large-caps, adjusted close, 10+ years; daily returns on common trading days |
| Symbolization | Per-stock or **pooled** training-window quantiles (20/40/60/80) → 5 symbols (big-down … big-up); optional market-neutral |
| Alignment | **Banded Needleman-Wunsch**, graded matrix (`+2` match, else `−|ord(s)−ord(t)|`), affine gaps (−1.0 / −0.4), band w = 5 |
| Distance | `1 − score/√(self_i·self_j)`; asserted symmetric + non-negative (**not** claimed to be a metric) |
| Clustering | Hierarchical **average linkage**; silhouette over k ∈ [2, min(6, ⌊N/3⌋)], **min cluster size 3** |
| Gate | Real silhouette vs 95th percentile of **circular-shift** nulls (same k selection); FDR a diagnostic only |
| Forecast | Per-cluster ARIMA, AIC order (p,q ≤ 2), d by ADF, **~21-day cumulative**; a modest tilt |
| Portfolio | **Two-stage**: across clusters (equal / risk-parity / tilt), within (equal / inverse-vol), cap `max(10%, 1.5/N)` |
| Rebalance | Monthly (21 days), walk-forward, 252-day training window |
| Costs | turnover = Σ\|Δw\| (total traded weight), **10 bps**, swept 0–25 bps |

## 3. Rigour controls

- **No look-ahead:** every stage at *t* uses data ≤ *t* only (quantiles, fits, clusters).
- **No data snooping:** sweeps tuned on the first ~70% of history, reported on the untouched last ~30%.
- **Paired** block bootstrap (same resampled blocks for method and baseline), 5,000 resamples.
- **Controls that isolate the method:** correlation clustering, **random clusters of the same sizes** through the same allocator, and sector clusters.
- **Replication checks:** every in-sample winner re-tested on a different period.
- **Pre-registration:** two confirmatory tests frozen before their data was fetched.
- **Sensitivity:** Politis–White automatic block length checked against the fixed 20 days.

## 4. Results, phase by phase

| Phase | Setup | Headline |
|---|---|---|
| Main run | original 20, report 2022–24, 31 rebalances | Alignment **worst of 7 methods** (Sharpe 0.33 vs equal-weight 0.75); significantly worse than equal-weight, mean-variance, random and sector (CIs exclude 0) |
| Isolation | same allocator, correlation distance | Correlation clustering 0.51–0.53 vs alignment 0.33–0.38 → the *distance* was the weak link |
| Portfolio sweep | risk-parity / inverse-vol | Concentration explains only +0.05 Sharpe; still well below equal weight. **Found and fixed a bug**: `risk_parity` was silently ignored without forecasts |
| Ablations (report) | 6 symbolizations | `per_stock_mnl` best (0.869 > 0.746); magnitude-aware (pooled) helps |
| Ablations (tune) | same 6, different period | **Ranking flips** — `per_stock_mnl` falls below equal weight. In-sample win was selection |
| **Holdout (pre-registered)** | disjoint 20 stocks, 131 rebalances | `pooled_mnl` 1.059 vs equal 1.039; CI **(-0.064, 0.107)** → **H1 not supported** |
| **Time OOS (pre-registered)** | original 20, 2005–2012, 83 rebalances | `pooled_mnl` 0.707 vs equal 0.639; CI **(-0.006, 0.141)** → **H2 not supported** |
| PW sensitivity | automatic block length (42, 52) | CIs essentially unchanged; conclusions robust |

**The consistent pattern:** `pooled_mnl` (magnitude-aware, market-neutral) beats equal
weight on Sharpe in **all four** evaluations — by +0.02, +0.02, +0.03, +0.07 — but **never
significantly**; every CI includes 0.

## 5. Verdict

**Neither a win nor a disproof — an underpowered, directionally positive null.**

- The pipeline is correct, reproducible and bias-aware; the alignment machinery works as designed.
- Under the baseline (per-stock, magnitude-discarding) symbolization the method is **significantly worse** than equal weight — so the obvious configuration is a genuine failure, not noise.
- The best configuration shows a **small, consistent, non-significant** edge across four evaluations.
- Claiming a win would be over-reach; claiming the method is worthless would also be over-reach.

The honest deliverable is the **methodology**: sequence alignment applied to return
patterns is implementable, testable, and here it is tested hard enough to know that the
naive version fails and the improved version is unproven.

## 6. Caveats

- **Survivorship bias:** fixed universes of survivors; documented, not corrected.
- **Underpowered:** 31–131 rebalances per test; CIs are wide by construction.
- **Not independent:** the four evaluations overlap in universe and time; no meta-analysis was attempted (it would over-claim).
- **The null gate is weak here:** it passes ~90% of rebalances because equities share a market factor — it detects "similarity exists", not "clusters are meaningful".
- **Clusters are band-sensitive:** ARI 0.07–0.40 across w ∈ {2, 5, 10}; a single band is not a robust specification.
- **Estimated costs only:** 10 bps flat, no market impact, no borrow.
- **Forecast near-noise:** ARIMA on daily returns ≈ the mean; the tilt is minor, and the headline results do not depend on it (default allocator is untitled).

## 7. What would change the conclusion

1. **More power:** more stocks, longer history, multiple markets — the effect size is small enough that only more data can resolve it.
2. **Fix the specification search:** pre-register one symbolization/allocator and test on genuinely fresh data (the two tests here are the template).
3. **Better clustering signal:** addressing the band sensitivity (e.g. consensus across bands) before any performance claim.

## 8. Reproduce

```bash
uv sync
uv run pytest                                    # 68 tests
uv run python -m src.backtest                    # main run (downloads 20 tickers)
uv run python scripts/run_full_report.py         # writes reports/metrics.csv etc.
uv run python scripts/run_experiments.py         # portfolio sweep + isolation
uv run python scripts/run_symbolize_ablations.py report   # also: tune
uv run python scripts/run_holdout_eval.py        # pre-registered, ticker-disjoint
uv run python scripts/run_time_oos.py            # pre-registered, time-disjoint
```

All parameters live in `config/params.yaml`; every experiment is seeded and reproducible.