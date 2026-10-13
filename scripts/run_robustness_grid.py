"""Robustness grid (M11): alignment Sharpe across band widths x allocator choices.

Cheap because the null gate is bypassed (it passes ~90%); each cell clusters once per
rebalance at the given band. Writes ``reports/robustness_grid.csv`` plus a cumulative
returns figure.

Usage::

    uv run python scripts/run_robustness_grid.py
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from genomefolio.align import distance_matrix
from genomefolio.backtest import performance_metrics
from genomefolio.cluster import select_constrained_k
from genomefolio.config import load_config
from genomefolio.costs import transaction_cost
from genomefolio.data import align_to_common_days, get_prices, time_split, to_returns
from genomefolio.evaluation import deflated_sharpe_from_returns
from genomefolio.plots import cumulative_returns_plot
from genomefolio.portfolio import equal_weights, two_stage_weights
from genomefolio.reporting import save_series, write_run_meta
from genomefolio.rolling import walk_forward_windows
from genomefolio.symbolize import symbolize

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
LOG = REPORTS / "robustness_grid.log"
COST_BPS = 10.0

ALLOCATORS = [("equal", "equal"), ("risk_parity", "inverse_vol"), ("risk_parity_lw", "inverse_vol_lw")]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _build(windows, tickers, cfg, band, pcfg, prev0):
    parts, prev = [], prev0
    for _, _, window, forward in windows:
        symbols = symbolize(window, cfg.symbolize.cutoffs, quantile_scope=cfg.symbolize.quantile_scope)
        D = distance_matrix(symbols, cfg.align, band=band)
        res = select_constrained_k(D, cfg.cluster, cfg.cluster.linkage)
        w = two_stage_weights(res.clusters, window, pcfg, None)
        if w.sum() <= 0:
            w = equal_weights(tickers)
        w = w.reindex(tickers).fillna(0.0)
        w = w / w.sum()
        turn = float((w - prev).abs().sum())
        prev = w
        seg = (forward * w).sum(axis=1).copy()
        seg.iloc[0] -= transaction_cost(turn, COST_BPS)
        parts.append(seg)
    return pd.concat(parts).sort_index()


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    cfg = load_config()
    returns = align_to_common_days(to_returns(get_prices(cfg)))
    _, report = time_split(returns, cfg.split.tune_fraction, enabled=cfg.split.enabled)
    target = report if len(report) else returns
    tickers = list(target.columns)
    windows = list(walk_forward_windows(target, cfg.window.train_days, cfg.window.rebalance_days))
    log(f"report days={len(target)} tickers={len(tickers)} rebalances={len(windows)}")

    rows, daily = {}, {}
    for band in cfg.align.band_sweep:
        for cm, wc in ALLOCATORS:
            pcfg = replace(cfg.portfolio, cluster_method=cm, within_cluster=wc)
            series = _build(windows, tickers, cfg, band, pcfg, pd.Series(0.0, index=tickers))
            key = f"band{band}_{cm}_{wc}"
            daily[key] = series
            m = performance_metrics(series)
            m["dsr"] = deflated_sharpe_from_returns(series, n_trials=len(cfg.align.band_sweep) * len(ALLOCATORS))["dsr"]
            rows[key] = m
            log(f"{key}: sharpe={m['sharpe']:.3f} dsr={m['dsr']:.3f}")

    pd.DataFrame(rows).T.round(6).to_csv(REPORTS / "robustness_grid.csv")
    save_series(daily, REPORTS / "robustness_grid_daily.csv")
    cumulative_returns_plot(daily, REPORTS / "figures" / "robustness_grid.png", "Robustness grid: cumulative returns")
    write_run_meta(REPORTS / "robustness_grid_meta.json", cfg)
    log("wrote robustness_grid.csv, robustness_grid_daily.csv, figures/robustness_grid.png")


if __name__ == "__main__":
    main()
