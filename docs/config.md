# Configuration

Every parameter lives in `config/params.yaml` (loaded by `genomefolio/config.py` into
validated frozen dataclasses; nothing is hardcoded). `config/sectors.yaml` and
`config/sectors_alt.yaml` map tickers to sector groups for the sector control.

| Key | Meaning |
|---|---|
| `seed` | master seed (makes every run reproducible) |
| `universe.tickers/start/end` | the fixed stock universe and date range |
| `universe.cache_dir` | where downloaded prices are cached (gitignored) |
| `universe.fixture_path` | committed offline sample |
| `universe.sectors_path` | sector map for the sector control |
| `split.enabled/tune_fraction` | chronological tune/report split |
| `symbolize.cutoffs` | quantile cutoffs (4/6/8 → 5/7/9 symbols) |
| `symbolize.quantile_scope` | `per_stock` (magnitude-normalised) or `pooled` (magnitude-aware) |
| `symbolize.market_neutral` | symbolise factor-residual returns |
| `align.match_score / gap_open / gap_extend / band` | alignment scoring |
| `align.band_sweep` | band widths for the stability sweep / consensus |
| `cluster.linkage / k_min / k_max / k_cap_divisor / min_cluster_size` | clustering |
| `null_test.gate_percentile / n_draws_tuning / n_draws_report / fdr_alpha` | gate |
| `window.train_days / rebalance_days / forecast_horizon` | walk-forward windows |
| `forecast.model / max_p / max_q / seasonal / fallback` | per-cluster forecast |
| `portfolio.cluster_method / within_cluster` | allocator (`equal`, `risk_parity[_lw]`, `forecast_tilt`; `equal`, `inverse_vol[_lw]`) |
| `portfolio.cap_floor / cap_factor` | per-stock cap `max(cap_floor, cap_factor / N)` |
| `portfolio.mv_risk_aversion / mean_shrink / forecast_tilt_strength` | optimiser knobs |
| `costs.base_bps / spread_bps / impact_coeff / sweep_bps` | transaction costs |
| `bootstrap.block_length / use_politis_white / n_resamples / confidence` | CIs |

Invalid or inconsistent values raise at load time (`load_config`), so misconfiguration
fails fast rather than mid-backtest.
