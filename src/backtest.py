"""Walk-forward backtest engine, metrics, and paired block bootstrap.

For each rebalance date the engine uses only the trailing training window
(``window.train_days``), builds the alignment (or a baseline) allocation, holds it over
the next ``window.rebalance_days``, and records the realised daily returns. Turnover is
``sum |dw|`` (total traded weight, buys plus sells) and cost is ``bps x turnover``.
Cluster churn between rebalances is therefore charged.

Time-series confidence intervals use a **paired block bootstrap**: one set of resampled
blocks is applied to both the method and its baseline, so differences are computed on
identical resamples. Block length is configurable (~20 days default) or chosen by the
Politis-White rule.

No look-ahead: symbol quantiles, the null gate, clusters and forecasts at time ``t`` all
use the window ending at ``t`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config, load_config, load_sectors
from src.portfolio import (
    correlation_clusters,
    equal_weights,
    optimised_weights,
    random_clusters,
    sector_clusters,
    two_stage_weights,
)

PERIODS_PER_YEAR = 252


@dataclass
class BacktestResult:
    method: str
    period_returns: list = field(default_factory=list)  # (rebalance_date, gross daily Series)
    turnover: pd.Series = field(default_factory=pd.Series)
    gate_passed: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    cost_bps: float = 0.0

    def net_returns(self, cost_bps: float | None = None) -> pd.Series:
        """Daily portfolio returns net of ``cost_bps`` on turnover, applied at each rebalance."""
        bps = self.cost_bps if cost_bps is None else cost_bps
        parts = []
        for date, period in self.period_returns:
            p = period.copy()
            p.iloc[0] -= (bps / 1e4) * float(self.turnover.get(date, 0.0))
            parts.append(p)
        return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)

    @property
    def daily_returns(self) -> pd.Series:
        return self.net_returns()

    @property
    def gate_pass_fraction(self) -> float:
        if not self.gate_passed:
            return float("nan")
        return float(np.mean(list(self.gate_passed.values())))


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def performance_metrics(daily: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> dict:
    """Annualized return, volatility, Sharpe, max drawdown, and total turnover-free stats."""
    daily = daily.dropna()
    if daily.empty:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    ann_ret = float((1.0 + daily).prod() ** (periods_per_year / len(daily)) - 1.0)
    sd = float(daily.std(ddof=0))
    ann_vol = float(sd * np.sqrt(periods_per_year))
    sharpe = float(daily.mean() / sd * np.sqrt(periods_per_year)) if sd > 1e-12 else np.nan
    cum = (1.0 + daily).cumprod()
    drawdown = cum / cum.cummax() - 1.0
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def _sharpe(daily: np.ndarray) -> float:
    sd = float(np.std(daily, ddof=0))
    return float(np.mean(daily) / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 1e-12 else np.nan


def _ann_return(daily: np.ndarray) -> float:
    return float(np.prod(1.0 + daily) ** (PERIODS_PER_YEAR / len(daily)) - 1.0)


def _politis_white_block_length(daily: np.ndarray) -> int:
    """Crude automatic block length; falls back to sqrt(n) when undetermined."""
    n = len(daily)
    if n < 20:
        return max(1, int(np.sqrt(n)))
    # Simple rule-of-thumb using the first-order autocorrelation.
    x = daily - daily.mean()
    denom = float(x @ x)
    rho = float(x[:-1] @ x[1:] / denom) if denom > 0 else 0.0
    if abs(rho) < 1e-6:
        return max(1, int(np.sqrt(n)))
    candidate = int(np.ceil((2.0 * abs(rho) / (1.0 - abs(rho))) ** (2.0 / 3.0) * n ** (1.0 / 3.0)))
    return int(np.clip(candidate, 1, n // 4 or 1))


def paired_block_bootstrap(
    method_daily: pd.Series,
    baseline_daily: pd.Series,
    n_resamples: int,
    block_length: int,
    confidence: float,
    rng: np.random.Generator,
    use_politis_white: bool = False,
) -> dict:
    """Paired block-bootstrap CIs on Sharpe and on Sharpe/return differences.

    Returns a dict with CI bounds for the method's Sharpe, the baseline's Sharpe, the
    Sharpe difference and the annualized-return difference (method - baseline).
    """
    common = method_daily.index.intersection(baseline_daily.index)
    m = method_daily.reindex(common).to_numpy(dtype=float)
    b = baseline_daily.reindex(common).to_numpy(dtype=float)
    n = len(m)
    if n < 10:
        nan = (np.nan, np.nan)
        return {"sharpe_method": nan, "sharpe_baseline": nan, "sharpe_diff": nan, "return_diff": nan}

    L = _politis_white_block_length(m) if use_politis_white else block_length
    L = int(np.clip(L, 1, n))
    n_blocks = int(np.ceil(n / L))
    starts_all = np.arange(0, n - L + 1)

    sharpe_m = np.empty(n_resamples)
    sharpe_b = np.empty(n_resamples)
    ret_m = np.empty(n_resamples)
    ret_b = np.empty(n_resamples)
    for r in range(n_resamples):
        starts = rng.choice(starts_all, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
        mm, bb = m[idx], b[idx]
        sharpe_m[r] = _sharpe(mm)
        sharpe_b[r] = _sharpe(bb)
        ret_m[r] = _ann_return(mm)
        ret_b[r] = _ann_return(bb)

    lo = (1.0 - confidence) / 2.0 * 100.0
    hi = (1.0 + confidence) / 2.0 * 100.0
    ci = lambda a: (float(np.nanpercentile(a, lo)), float(np.nanpercentile(a, hi)))  # noqa: E731
    return {
        "sharpe_method": ci(sharpe_m),
        "sharpe_baseline": ci(sharpe_b),
        "sharpe_diff": ci(sharpe_m - sharpe_b),
        "return_diff": ci(ret_m - ret_b),
    }


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------
def _rebalance_positions(n: int, train_days: int, rebalance_days: int) -> list[int]:
    from src.rolling import rebalance_positions

    return rebalance_positions(n, train_days, rebalance_days)


def _alignment_clusters(symbols, cfg: Config):
    from src.align import distance_matrix
    from src.cluster import select_constrained_k

    D = distance_matrix(symbols, cfg.align)
    return select_constrained_k(D, cfg.cluster, cfg.cluster.linkage)


def _alignment_weights(window, cfg: Config, rng: np.random.Generator, n_draws: int, gate_record: dict, date):
    from src.forecast import forecast_clusters
    from src.null_test import evaluate_gate
    from src.symbolize import symbolize

    symbols = symbolize(
        window,
        cfg.symbolize.cutoffs,
        market_neutral=cfg.symbolize.market_neutral,
        quantile_scope=cfg.symbolize.quantile_scope,
    )
    gate = evaluate_gate(symbols, cfg.align, cfg.cluster, cfg.null_test, n_draws=n_draws, rng=rng)
    gate_record[date] = gate.passed
    if not gate.passed:
        return equal_weights(window.columns), gate.real_result.clusters
    clusters = gate.real_result.clusters
    composite = pd.DataFrame(
        {c: window[list(members)].mean(axis=1) for c, members in clusters.items()}
    )
    forecasts = {
        c: fc.expected_return
        for c, fc in forecast_clusters(composite, cfg.forecast, cfg.window.forecast_horizon).items()
    }
    return two_stage_weights(clusters, window, cfg.portfolio, forecasts), clusters


def run_backtest(
    returns: pd.DataFrame,
    cfg: Config,
    method: str = "alignment",
    n_draws: int | None = None,
    cost_bps: float | None = None,
    rng: np.random.Generator | None = None,
) -> BacktestResult:
    """Run one walk-forward backtest for ``method`` over ``returns``."""
    rng = rng or np.random.default_rng(cfg.seed)
    n_draws = n_draws if n_draws is not None else cfg.null_test.n_draws_report
    cost_bps = cfg.costs.base_bps if cost_bps is None else cost_bps
    date_rng = np.random.default_rng(cfg.seed)
    sectors = load_sectors()

    positions = _rebalance_positions(len(returns), cfg.window.train_days, cfg.window.rebalance_days)
    tickers = list(returns.columns)
    cap = cfg.portfolio.cap(len(tickers))

    prev_w = pd.Series(0.0, index=tickers)
    period_returns: list = []
    turnover_records: dict = {}
    gate_record: dict = {}
    weight_record: dict = {}

    for step, i in enumerate(positions):
        window = returns.iloc[i - cfg.window.train_days : i]
        forward = returns.iloc[i : i + cfg.window.rebalance_days]
        date = returns.index[i]

        if method == "alignment":
            w, _ = _alignment_weights(
                window, cfg, np.random.default_rng(cfg.seed + step), n_draws, gate_record, date
            )
        elif method == "equal_weight":
            w = equal_weights(tickers)
        elif method in {"min_variance", "mean_variance"}:
            w = optimised_weights(
                window, cap, risk_aversion=0.5 if method == "mean_variance" else 0.0,
                shrink_means=(method == "mean_variance"),
            )
        elif method == "correlation":
            clusters = correlation_clusters(window, cfg.cluster)
            w = two_stage_weights(clusters, window, cfg.portfolio)
        elif method == "sector":
            clusters = sector_clusters(sectors, tickers)
            w = two_stage_weights(clusters, window, cfg.portfolio)
        elif method == "random":
            from src.symbolize import symbolize

            clusters = _alignment_clusters(
                symbolize(window, cfg.symbolize.cutoffs, quantile_scope=cfg.symbolize.quantile_scope), cfg
            ).clusters
            sizes = [len(clusters[c]) for c in sorted(clusters)]
            rclusters = random_clusters(sizes, tickers, date_rng)
            w = two_stage_weights(rclusters, window, cfg.portfolio)
        else:  # pragma: no cover
            raise ValueError(f"unknown method {method}")

        w = w.reindex(tickers).fillna(0.0)

        turnover = float((w - prev_w).abs().sum())
        turnover_records[date] = turnover
        weight_record[date] = w.copy()
        prev_w = w

        period = (forward * w).sum(axis=1)
        period_returns.append((date, period))

    turnovers = pd.Series(turnover_records)
    return BacktestResult(
        method=method,
        period_returns=period_returns,
        turnover=turnovers,
        gate_passed=gate_record,
        weights=weight_record,
        cost_bps=cost_bps,
    )


def metrics_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Metrics for each method on identical daily windows."""
    rows = {}
    for name, res in results.items():
        m = performance_metrics(res.daily_returns)
        m["avg_turnover"] = float(res.turnover.mean()) if len(res.turnover) else np.nan
        if res.gate_passed:
            m["gate_pass_fraction"] = res.gate_pass_fraction
        rows[name] = m
    return pd.DataFrame(rows).T


def gate_passed_only_view(results: dict[str, BacktestResult], reference: str = "alignment") -> pd.DataFrame:
    """Descriptive view restricted to rebalances where the alignment gate passed.

    Every baseline is restricted to the **same** rebalances; this is descriptive only
    (no CI), because non-contiguous months break the block structure.
    """
    ref = results[reference]
    keep = [d for d, ok in ref.gate_passed.items() if ok]
    rows = {}
    for name, res in results.items():
        parts = []
        for d, p in res.period_returns:
            if d in keep:
                pp = p.copy()
                pp.iloc[0] -= (res.cost_bps / 1e4) * float(res.turnover.get(d, 0.0))
                parts.append(pp)
        daily = pd.concat(parts) if parts else pd.Series(dtype=float)
        m = performance_metrics(daily)
        m["avg_turnover"] = float(res.turnover.reindex(keep).mean()) if keep else np.nan
        rows[name] = m
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------------------
# Reporting / CLI
# --------------------------------------------------------------------------------------
def band_sweep_report(returns: pd.DataFrame, cfg: Config) -> dict:
    """Cluster once on the latest training window across band widths; report stability."""
    from src.cluster import band_sweep, stability_ari
    from src.symbolize import symbolize

    window = returns.iloc[-cfg.window.train_days :]
    symbols = symbolize(
        window,
        cfg.symbolize.cutoffs,
        market_neutral=cfg.symbolize.market_neutral,
        quantile_scope=cfg.symbolize.quantile_scope,
    )
    sweep = band_sweep(symbols, cfg.align, cfg.cluster, cfg.align.band_sweep)
    return {
        "n_clusters": {b: r.n_clusters for b, r in sweep.items()},
        "silhouette": {b: r.silhouette for b, r in sweep.items()},
        "ari": stability_ari(sweep),
    }


def run_full_report(
    returns: pd.DataFrame,
    cfg: Config,
    methods: tuple[str, ...],
    n_draws: int,
    cost_bps: float | None = None,
) -> dict:
    """Run all methods on identical windows plus paired-bootstrap CIs vs baselines."""
    results = {m: run_backtest(returns, cfg, method=m, n_draws=n_draws, cost_bps=cost_bps) for m in methods}
    table = metrics_table(results)
    descriptive = gate_passed_only_view(results, reference="alignment") if results["alignment"].gate_passed else None

    cis = {}
    if "alignment" in results:
        rng = np.random.default_rng(cfg.seed + 999)
        for name, res in results.items():
            if name == "alignment":
                continue
            cis[name] = paired_block_bootstrap(
                results["alignment"].daily_returns,
                res.daily_returns,
                cfg.bootstrap.n_resamples,
                cfg.bootstrap.block_length,
                cfg.bootstrap.confidence,
                rng,
                cfg.bootstrap.use_politis_white,
            )
    return {"results": results, "table": table, "descriptive": descriptive, "cis": cis}


def _print_report(report: dict, cfg: Config) -> None:
    pd.set_option("display.width", 160)
    print("\n=== Metrics (all methods, identical windows) ===")
    print(report["table"].round(4).to_string())
    if report.get("descriptive") is not None:
        print("\n=== Alignment gate-passed-only (descriptive, no CI; baselines on same rebalances) ===")
        print(report["descriptive"].round(4).to_string())
    if report["cis"]:
        print("\n=== Paired block-bootstrap 95% CIs (alignment vs baseline) ===")
        for name, ci in report["cis"].items():
            print(f"  {name:14s} sharpe_diff={ci['sharpe_diff']!s}  return_diff={ci['return_diff']!s}")


def main(config_path, use_fixture, n_draws, cost_bps, methods, band_only):
    """Entry point shared by the CLI."""
    from src.data import align_to_common_days, get_prices, load_fixture, time_split, to_returns

    cfg = load_config(config_path)
    prices = load_fixture(cfg) if use_fixture else get_prices(cfg)
    returns = align_to_common_days(to_returns(prices))
    tune, report = time_split(returns, cfg.split.tune_fraction, enabled=cfg.split.enabled and not use_fixture)
    target = report if len(report) else returns
    print(f"Universe: {len(returns.columns)} stocks | days={len(returns)} | train={cfg.window.train_days} "
          f"| tune={len(tune)} report={len(target)} | draws={n_draws}")

    sweep = band_sweep_report(target, cfg)
    print("\n=== Band-sweep stability (w -> n_clusters, silhouette) ===")
    for b in sorted(sweep["n_clusters"]):
        print(f"  w={b:2d}: k={sweep['n_clusters'][b]}  silhouette={sweep['silhouette'][b]:.3f}")
    print("ARI across bands:\n" + sweep["ari"].round(3).to_string())
    if band_only:
        return

    report_out = run_full_report(target, cfg, tuple(methods), n_draws=n_draws, cost_bps=cost_bps)
    _print_report(report_out, cfg)

    gate = report_out["results"]["alignment"]
    print(f"\nAlignment gate pass fraction: {gate.gate_pass_fraction:.3f} "
          f"(~0.05 expected by chance at the {cfg.null_test.gate_percentile:.0f}th percentile)")

    print("\n=== Transaction-cost sweep (alignment) ===")
    align_res = report_out["results"]["alignment"]
    for bps in cfg.costs.sweep_bps:
        m = performance_metrics(align_res.net_returns(bps))
        print(f"  {bps:5.1f} bps: ann_return={m['ann_return']:+.4f}  sharpe={m['sharpe']:+.3f}")


if __name__ == "__main__":  # pragma: no cover
    import click

    @click.command()
    @click.option("--config", "config_path", default=None, help="Path to params.yaml")
    @click.option("--use-fixture", is_flag=True, help="Use the committed offline sample")
    @click.option("--draws", "n_draws", default=None, type=int, help="Null-gate draws")
    @click.option("--cost-bps", default=None, type=float, help="Override transaction cost (bps)")
    @click.option(
        "--methods",
        default="alignment,equal_weight,min_variance,mean_variance,correlation,random,sector",
    )
    @click.option("--band-only", is_flag=True, help="Only print the band-sweep stability")
    def _cli(config_path, use_fixture, n_draws, cost_bps, methods, band_only):
        cfg = load_config(config_path)
        main(
            config_path,
            use_fixture,
            n_draws if n_draws is not None else cfg.null_test.n_draws_report,
            cost_bps,
            tuple(m.strip() for m in methods.split(",") if m.strip()),
            band_only,
        )

    _cli()
