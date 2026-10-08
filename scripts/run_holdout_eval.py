"""Pre-registered holdout evaluation on a fresh universe.

See ``reports/PREREGISTRATION.md`` (written before this was run). One confirmatory test:
``pooled_mnl`` vs equal weight on a disjoint 20-stock universe.

The null gate is bypassed (frozen in the pre-registration), consistent with the ablation
study, so the numbers are comparable and the run is fast.

Usage::

    uv run python scripts/run_holdout_eval.py
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.align import distance_matrix
from src.backtest import paired_block_bootstrap, performance_metrics
from src.cluster import select_constrained_k
from src.config import load_config
from src.data import align_to_common_days, get_prices, to_returns
from src.portfolio import equal_weights, two_stage_weights
from src.rolling import walk_forward_windows
from src.symbolize import symbolize

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
LOG = REPORTS / "holdout.log"
COST_BPS = 10.0
BASE_CUT = (0.20, 0.40, 0.60, 0.80)

# frozen in reports/PREREGISTRATION.md — do not edit
HOLDOUT_TICKERS = (
    "NVDA", "AMD", "QCOM", "TXN", "ORCL", "ADBE", "CRM", "IBM", "ABT", "BMY",
    "LLY", "AMGN", "GILD", "TGT", "LOW", "SBUX", "BA", "GE", "MMM", "HON",
)
HOLDOUT_START, HOLDOUT_END = "2013-01-01", "2025-01-01"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _series(windows, tickers, cfg, portfolio_cfg, cutoffs, mn, scope):
    parts, prev = [], pd.Series(0.0, index=tickers)
    for _, _, window, forward in windows:
        symbols = symbolize(window, cutoffs, market_neutral=mn, quantile_scope=scope)
        D = distance_matrix(symbols, cfg.align)
        res = select_constrained_k(D, cfg.cluster, cfg.cluster.linkage)
        w = two_stage_weights(res.clusters, window, portfolio_cfg).reindex(tickers).fillna(0.0)
        turn = float((w - prev).abs().sum())
        prev = w
        seg = (forward * w).sum(axis=1).copy()
        seg.iloc[0] -= (COST_BPS / 1e4) * turn
        parts.append(seg)
    return pd.concat(parts).sort_index()


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    base = load_config()
    uni = replace(
        base.universe,
        tickers=HOLDOUT_TICKERS,
        start=HOLDOUT_START,
        end=HOLDOUT_END,
        cache_dir="data/cache_holdout",
    )
    cfg = replace(base, universe=uni)
    prices = get_prices(cfg)
    returns = align_to_common_days(to_returns(prices))
    tickers = list(returns.columns)
    windows = list(walk_forward_windows(returns, cfg.window.train_days, cfg.window.rebalance_days))
    pcfg = replace(cfg.portfolio, cluster_method="risk_parity", within_cluster="inverse_vol")
    log(f"holdout tickers={len(tickers)} days={len(returns)} rebalances={len(windows)}")

    daily = {}
    daily["per_stock"] = _series(windows, tickers, cfg, pcfg, BASE_CUT, False, "per_stock")
    daily["pooled_mnl"] = _series(windows, tickers, cfg, pcfg, BASE_CUT, True, "pooled")

    parts, prev = [], pd.Series(0.0, index=tickers)
    for _, _, _, forward in windows:
        w = equal_weights(tickers)
        turn = float((w - prev).abs().sum())
        prev = w
        seg = (forward * w).sum(axis=1).copy()
        seg.iloc[0] -= (COST_BPS / 1e4) * turn
        parts.append(seg)
    daily["equal_weight"] = pd.concat(parts).sort_index()

    table = {name: performance_metrics(s) for name, s in daily.items()}
    pd.DataFrame(table).T.to_csv(REPORTS / "holdout_eval.csv")
    for name, m in table.items():
        log(f"{name}: sharpe={m['sharpe']:.3f} ann_return={m['ann_return']:.4f} vol={m['ann_vol']:.3f}")

    rng = np.random.default_rng(base.seed + 999)
    ci = {
        name: paired_block_bootstrap(
            daily[name], daily["equal_weight"],
            base.bootstrap.n_resamples, base.bootstrap.block_length,
            base.bootstrap.confidence, rng, base.bootstrap.use_politis_white,
        )
        for name in ("per_stock", "pooled_mnl")
    }
    pd.DataFrame(ci).T.to_csv(REPORTS / "holdout_ci.csv")
    log(f"pooled_mnl sharpe diff vs equal_weight: {ci['pooled_mnl']['sharpe_diff']}")
    lo, hi = ci["pooled_mnl"]["sharpe_diff"]
    if lo > 0 and hi > 0:
        verdict = "H1 SUPPORTED"
    elif hi < 0:
        verdict = "H1 FAILED (significantly worse)"
    else:
        verdict = "H1 NOT SUPPORTED (CI includes 0)"
    log(verdict)
    log("wrote holdout_eval.csv, holdout_ci.csv")


if __name__ == "__main__":
    main()
