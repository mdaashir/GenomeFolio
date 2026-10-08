"""Produce the full results report for the study.

Runs the band sweep, every method on identical windows, the paired block bootstrap,
the transaction-cost sweep, and the descriptive gate-passed-only view, writing each
table to ``reports/`` and a progress log to ``reports/run.log`` (flushed per step so the
run can be monitored even when stdout is not).

Usage::

    uv run python scripts/run_full_report.py [n_draws]

``n_draws`` defaults to ``null_test.n_draws_report`` from the config (the reported-run
budget; tuning sweeps use the smaller budget).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import (
    band_sweep_report,
    gate_passed_only_view,
    metrics_table,
    paired_block_bootstrap,
    performance_metrics,
    run_backtest,
)
from src.config import load_config
from src.data import align_to_common_days, get_prices, time_split, to_returns

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
LOG = REPORTS / "run.log"
METHODS = ("alignment", "equal_weight", "min_variance", "mean_variance", "correlation", "random", "sector")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main(n_draws: int) -> None:
    LOG.write_text("", encoding="utf-8")
    cfg = load_config()
    prices = get_prices(cfg)
    log(f"prices: {prices.shape[0]} days x {prices.shape[1]} tickers")
    returns = align_to_common_days(to_returns(prices))
    tune, report = time_split(returns, cfg.split.tune_fraction, enabled=cfg.split.enabled)
    target = report if len(report) else returns
    log(f"universe={returns.shape[1]} days={len(returns)} tune={len(tune)} report={len(target)} draws={n_draws}")

    sweep = band_sweep_report(target, cfg)
    log(f"band sweep: k={sweep['n_clusters']} silhouette={sweep['silhouette']}")
    sweep["ari"].round(4).to_csv(REPORTS / "band_sweep_ari.csv")

    results = {}
    for method in METHODS:
        t0 = time.perf_counter()
        results[method] = run_backtest(target, cfg, method=method, n_draws=n_draws)
        log(f"{method}: {time.perf_counter() - t0:.1f}s gate_pass={results[method].gate_pass_fraction:.3f}")

    table = metrics_table(results).round(6)
    table.to_csv(REPORTS / "metrics.csv")
    log("wrote metrics.csv")

    descriptive = gate_passed_only_view(results).round(6)
    descriptive.to_csv(REPORTS / "metrics_gate_passed_only.csv")
    log("wrote metrics_gate_passed_only.csv")

    rng = np.random.default_rng(cfg.seed + 999)
    ci = {
        m: paired_block_bootstrap(
            results["alignment"].daily_returns,
            results[m].daily_returns,
            cfg.bootstrap.n_resamples,
            cfg.bootstrap.block_length,
            cfg.bootstrap.confidence,
            rng,
            cfg.bootstrap.use_politis_white,
        )
        for m in METHODS
        if m != "alignment"
    }
    pd.DataFrame(ci).T.to_csv(REPORTS / "bootstrap_ci.csv")
    log("wrote bootstrap_ci.csv")

    cost = {bps: performance_metrics(results["alignment"].net_returns(bps)) for bps in cfg.costs.sweep_bps}
    pd.DataFrame(cost).T.round(6).to_csv(REPORTS / "cost_sweep.csv")
    log("wrote cost_sweep.csv")

    results["alignment"].turnover.to_frame("turnover").to_csv(REPORTS / "turnover.csv")
    log("done")


if __name__ == "__main__":
    cfg = load_config()
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else cfg.null_test.n_draws_report
    main(draws)
