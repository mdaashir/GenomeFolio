"""Power study across universes (M9): does the small edge over equal weight hold up?

Runs the confirmatory arms (``per_stock``, ``pooled_mnl``) and equal weight on several
universes — the original, the ticker-disjoint holdout, and the alternate fixture — and
reports Sharpe, deflated Sharpe and a paired CI for ``pooled_mnl`` vs equal weight.

Bounded and fast: the null gate is bypassed (it passes ~90%).

Usage::

    uv run python scripts/run_power_study.py
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from genomefolio.align import distance_matrix
from genomefolio.backtest import paired_block_bootstrap, performance_metrics, seed_for
from genomefolio.cluster import select_constrained_k
from genomefolio.config import load_config
from genomefolio.costs import transaction_cost
from genomefolio.data import align_to_common_days, get_prices, load_fixture, to_returns
from genomefolio.evaluation import deflated_sharpe_from_returns
from genomefolio.portfolio import equal_weights, two_stage_weights
from genomefolio.reporting import save_series, write_run_meta
from genomefolio.rolling import walk_forward_windows
from genomefolio.symbolize import symbolize

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
LOG = REPORTS / "power_study.log"
COST_BPS = 10.0
BASE_CUT = (0.20, 0.40, 0.60, 0.80)

HOLDOUT = (
    "NVDA", "AMD", "QCOM", "TXN", "ORCL", "ADBE", "CRM", "IBM", "ABT", "BMY",
    "LLY", "AMGN", "GILD", "TGT", "LOW", "SBUX", "BA", "GE", "MMM", "HON",
)
ALT = ("ORCL", "ADBE", "CRM", "QCOM", "TXN", "IBM", "ABT", "BMY", "LLY", "TGT", "LOW", "SBUX")
ARMS = {"per_stock": (False, "per_stock"), "pooled_mnl": (True, "pooled")}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _alignment_arm(returns, cfg, mn, scope):
    tickers = list(returns.columns)
    windows = walk_forward_windows(returns, cfg.window.train_days, cfg.window.rebalance_days)
    pcfg = replace(cfg.portfolio, cluster_method="risk_parity", within_cluster="inverse_vol")
    parts, prev = [], pd.Series(0.0, index=tickers)
    for _, _, window, forward in windows:
        symbols = symbolize(window, BASE_CUT, market_neutral=mn, quantile_scope=scope)
        D = distance_matrix(symbols, cfg.align)
        res = select_constrained_k(D, cfg.cluster, cfg.cluster.linkage)
        w = two_stage_weights(res.clusters, window, pcfg).reindex(tickers).fillna(0.0)
        turn = float((w - prev).abs().sum())
        prev = w
        seg = (forward * w).sum(axis=1).copy()
        seg.iloc[0] -= transaction_cost(turn, COST_BPS)
        parts.append(seg)
    return pd.concat(parts).sort_index()


def _equal_arm(returns, cfg):
    tickers = list(returns.columns)
    windows = walk_forward_windows(returns, cfg.window.train_days, cfg.window.rebalance_days)
    parts, prev = [], pd.Series(0.0, index=tickers)
    for _, _, _, forward in windows:
        w = equal_weights(tickers)
        turn = float((w - prev).abs().sum())
        prev = w
        seg = (forward * w).sum(axis=1).copy()
        seg.iloc[0] -= transaction_cost(turn, COST_BPS)
        parts.append(seg)
    return pd.concat(parts).sort_index()


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    base = load_config()
    universes = [
        ("original", base.universe, "get"),
        ("holdout", replace(base.universe, tickers=HOLDOUT, cache_dir="data/cache_holdout"), "get"),
        (
            "alt",
            replace(
                base.universe,
                tickers=ALT,
                start="2019-01-01",
                end="2024-01-01",
                fixture_path="data/sample/prices_alt.csv",
            ),
            "fixture",
        ),
    ]

    rows, daily, ci_rows = {}, {}, {}
    for label, uni, loader in universes:
        cfg = replace(base, universe=uni)
        prices = load_fixture(cfg) if loader == "fixture" else get_prices(cfg)
        r = align_to_common_days(to_returns(prices))
        log(f"{label}: tickers={r.shape[1]} days={len(r)}")
        n_trials = len(ARMS) * len(universes)
        for arm, (mn, scope) in ARMS.items():
            s = _alignment_arm(r, cfg, mn, scope)
            daily[f"{label}:{arm}"] = s
            m = performance_metrics(s)
            m["dsr"] = deflated_sharpe_from_returns(s, n_trials=n_trials)["dsr"]
            rows[f"{label}:{arm}"] = m
            log(f"  {arm}: sharpe={m['sharpe']:.3f} dsr={m['dsr']:.3f}")
        eq = _equal_arm(r, cfg)
        daily[f"{label}:equal_weight"] = eq
        rows[f"{label}:equal_weight"] = performance_metrics(eq)
        ci_rows[label] = paired_block_bootstrap(
            daily[f"{label}:pooled_mnl"],
            eq,
            base.bootstrap.n_resamples,
            base.bootstrap.block_length,
            base.bootstrap.confidence,
            np.random.default_rng(seed_for(f"power_{label}", base.seed)),
            base.bootstrap.use_politis_white,
        )

    pd.DataFrame(rows).T.round(6).to_csv(REPORTS / "power_study.csv")
    pd.DataFrame(ci_rows).T.to_csv(REPORTS / "power_study_ci.csv")
    save_series(daily, REPORTS / "power_study_daily.csv")
    write_run_meta(REPORTS / "power_study_meta.json", base, {"universes": [u[0] for u in universes]})
    log("wrote power_study.csv, power_study_ci.csv, power_study_daily.csv, power_study_meta.json")


if __name__ == "__main__":
    main()
