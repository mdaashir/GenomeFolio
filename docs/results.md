# Results

The full study lives in `reports/`:

| Report | Contents |
|---|---|
| `REPORT.md` | consolidated study (question, method, results, caveats) |
| `RESULTS.md` | main run tables and interpretation |
| `EXPERIMENTS.md` | portfolio sweep + alignment isolation |
| `ABLATIONS.md` | symbolization ablations (two periods) |
| `PREREGISTRATION.md` / `HOLDOUT.md` | ticker-disjoint confirmatory test |
| `PREREGISTRATION_TIME.md` / `TIME_OOS.md` | time-disjoint confirmatory test |
| `PREREGISTRATION_SMOKE` (json) | the generic pre-registration harness example |
| `power_study.csv`, `robustness_grid.csv` | cross-universe and band x allocator results |

## Headline

Both pre-registered confirmatory tests **did not support** a beat over equal weight:

- Holdout universe: `pooled_mnl` Sharpe difference vs equal weight `(-0.064, 0.107)`.
- Time-disjoint (2005–2012): `(-0.006, 0.141)`.

`pooled_mnl` beats equal weight on Sharpe in all four evaluations (by +0.02 to +0.07) but
never significantly — a small, directionally consistent, **underpowered** effect. The naive
per-stock symbolization is significantly *worse* than equal weight, so the failure is real,
not noise. Deflated Sharpe across the robustness grid lands at 0.16–0.33, reflecting the
specification search.

See the reports for the full caveats (survivorship, short post-split period, single asset
class, selection risk).
