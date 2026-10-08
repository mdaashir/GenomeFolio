"""Follow-up experiments.

Two questions:

1. **Concentration hypothesis** — does the alignment portfolio's underperformance come
   from equal weight *across clusters* over-weighting small clusters? Sweeps the
   portfolio layer: risk-parity across clusters, inverse-vol within, and a forecast tilt.
2. **Isolate the alignment step** — same constrained-k clustering on plain correlation
   distance (``1 - corr``), to see whether sequence alignment adds anything over a
   correlation-based grouping.

The expensive null gate is computed **once per rebalance** and its clusters reused across
every portfolio variant, so the sweep is cheap.

Usage::

    uv run python scripts/run_experiments.py [n_draws]

Writes ``reports/experiments.csv``, ``reports/experiments_ci.csv`` and
``reports/experiments.log``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.align import distance_matrix
from src.backtest import paired_block_bootstrap, performance_metrics
from src.cluster import select_constrained_k
from src.config import load_config
from src.data import align_to_common_days, get_prices, time_split, to_returns
from src.forecast import forecast_clusters
from src.null_test import evaluate_gate
from src.portfolio import correlation_clusters, equal_weights, two_stage_weights
from src.rolling import walk_forward_windows
from src.symbolize import symbolize

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
LOG = REPORTS / "experiments.log"
COST_BPS = 10.0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _build_series(records, tickers, portfolio_cfg, source, use_forecasts):
    prev = pd.Series(0.0, index=tickers)
    parts, turnover = [], {}
    for rec in records:
        if source == "equal":
            w = equal_weights(tickers)
        else:
            clusters = rec["align"] if source == "align" else rec["corr"]
            if clusters is None:
                w = equal_weights(tickers)
            else:
                fc = rec["forecasts"] if use_forecasts else None
                w = two_stage_weights(clusters, rec["window"], portfolio_cfg, fc)
        w = w.reindex(tickers).fillna(0.0)
        turn = float((w - prev).abs().sum())
        turnover[rec["date"]] = turn
        prev = w
        period = (rec["forward"] * w).sum(axis=1).copy()
        period.iloc[0] -= (COST_BPS / 1e4) * turn
        parts.append(period)
    daily = pd.concat(parts).sort_index()
    return daily, pd.Series(turnover)


def main(n_draws: int) -> None:
    LOG.write_text("", encoding="utf-8")
    cfg = load_config()
    returns = align_to_common_days(to_returns(get_prices(cfg)))
    _, report = time_split(returns, cfg.split.tune_fraction, enabled=cfg.split.enabled)
    target = report if len(report) else returns
    tickers = list(target.columns)
    log(f"report days={len(target)} tickers={len(tickers)} draws={n_draws}")

    records = []
    t0 = time.perf_counter()
    for step, (_, date, window, forward) in enumerate(
        walk_forward_windows(target, cfg.window.train_days, cfg.window.rebalance_days)
    ):
        symbols = symbolize(window, cfg.symbolize.cutoffs, market_neutral=cfg.symbolize.market_neutral)
        gate = evaluate_gate(
            symbols, cfg.align, cfg.cluster, cfg.null_test, n_draws, np.random.default_rng(cfg.seed + step)
        )
        align_clusters = gate.real_result.clusters if gate.passed else None
        forecasts = None
        if align_clusters is not None:
            composite = pd.DataFrame({c: window[list(m)].mean(axis=1) for c, m in align_clusters.items()})
            forecasts = {
                c: fc.expected_return
                for c, fc in forecast_clusters(composite, cfg.forecast, cfg.window.forecast_horizon).items()
            }
        records.append(
            {
                "date": date,
                "window": window,
                "forward": forward,
                "align": align_clusters,
                "corr": correlation_clusters(window, cfg.cluster),
                "forecasts": forecasts,
                "passed": gate.passed,
            }
        )
        if (step + 1) % 5 == 0:
            log(f"rebalance {step + 1} gated ({time.perf_counter() - t0:.0f}s)")
    log(f"all rebalances gated in {time.perf_counter() - t0:.0f}s")

    variants = [
        ("equal_weight", "equal", "equal", "equal"),
        ("align_equal_equal", "align", "equal", "equal"),
        ("align_risparity_equal", "align", "risk_parity", "equal"),
        ("align_equal_invvol", "align", "equal", "inverse_vol"),
        ("align_risparity_invvol", "align", "risk_parity", "inverse_vol"),
        ("align_tilt_invvol", "align", "forecast_tilt", "inverse_vol"),
        ("corr_equal_equal", "corr", "equal", "equal"),
        ("corr_risparity_invvol", "corr", "risk_parity", "inverse_vol"),
    ]

    daily = {}
    metrics_rows = {}
    for name, source, cluster_method, within in variants:
        pcfg = replace(cfg.portfolio, cluster_method=cluster_method, within_cluster=within)
        use_fc = cluster_method == "forecast_tilt"
        d, turn = _build_series(records, tickers, pcfg, source, use_fc)
        daily[name] = d
        m = performance_metrics(d)
        m["avg_turnover"] = float(turn.mean())
        metrics_rows[name] = m
        log(f"{name}: sharpe={m['sharpe']:.3f} ann_return={m['ann_return']:.4f} turnover={m['avg_turnover']:.3f}")

    pd.DataFrame(metrics_rows).T.to_csv(REPORTS / "experiments.csv")

    rng = np.random.default_rng(cfg.seed + 999)
    ci = {
        name: paired_block_bootstrap(
            daily[name], daily["equal_weight"],
            cfg.bootstrap.n_resamples, cfg.bootstrap.block_length,
            cfg.bootstrap.confidence, rng, cfg.bootstrap.use_politis_white,
        )
        for name in daily
        if name != "equal_weight"
    }
    pd.DataFrame(ci).T.to_csv(REPORTS / "experiments_ci.csv")
    log("wrote experiments.csv and experiments_ci.csv")


if __name__ == "__main__":
    cfg = load_config()
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
