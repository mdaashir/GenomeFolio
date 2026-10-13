"""Time-disjoint out-of-sample evaluation (pre-2013).

See ``reports/PREREGISTRATION_TIME.md`` (written before the data was fetched). Freezes the
original 20-stock universe and evaluates on 2005-2012, an unseen era.

Reuses the arm builder from ``run_holdout_eval`` so the logic is identical to the
ticker-disjoint test.

Usage::

    uv run python scripts/run_time_oos.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_holdout_eval as h

from genomefolio.backtest import (
    paired_block_bootstrap,
    performance_metrics,
    politis_white_block_length,
    seed_for,
)
from genomefolio.config import load_config
from genomefolio.data import align_to_common_days, get_prices, to_returns
from genomefolio.portfolio import equal_weights
from genomefolio.reporting import save_series, write_run_meta
from genomefolio.rolling import walk_forward_windows

REPORTS = Path("reports")
LOG = REPORTS / "time_oos.log"
START, END = "2005-01-01", "2012-12-31"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    base = load_config()
    uni = replace(base.universe, start=START, end=END, cache_dir="data/cache_pretime")
    cfg = replace(base, universe=uni)
    prices = get_prices(cfg)
    returns = align_to_common_days(to_returns(prices))
    tickers = list(returns.columns)
    windows = list(walk_forward_windows(returns, cfg.window.train_days, cfg.window.rebalance_days))
    pcfg = replace(cfg.portfolio, cluster_method="risk_parity", within_cluster="inverse_vol")
    log(f"period={START}..{END} tickers={len(tickers)} days={len(returns)} rebalances={len(windows)}")

    daily = {
        "per_stock": h._series(windows, tickers, cfg, pcfg, h.BASE_CUT, False, "per_stock"),
        "pooled_mnl": h._series(windows, tickers, cfg, pcfg, h.BASE_CUT, True, "pooled"),
    }

    parts, prev = [], pd.Series(0.0, index=tickers)
    for _, _, _, forward in windows:
        w = equal_weights(tickers)
        turn = float((w - prev).abs().sum())
        prev = w
        seg = (forward * w).sum(axis=1).copy()
        seg.iloc[0] -= (h.COST_BPS / 1e4) * turn
        parts.append(seg)
    daily["equal_weight"] = pd.concat(parts).sort_index()

    table = {name: performance_metrics(s) for name, s in daily.items()}
    pd.DataFrame(table).T.to_csv(REPORTS / "time_oos_eval.csv")
    for name, m in table.items():
        log(f"{name}: sharpe={m['sharpe']:.3f} ann_return={m['ann_return']:.4f} vol={m['ann_vol']:.3f}")

    ci = {
        name: paired_block_bootstrap(
            daily[name], daily["equal_weight"],
            base.bootstrap.n_resamples, base.bootstrap.block_length,
            base.bootstrap.confidence, np.random.default_rng(seed_for(name, base.seed)),
            base.bootstrap.use_politis_white,
        )
        for name in ("per_stock", "pooled_mnl")
    }
    pd.DataFrame(ci).T.to_csv(REPORTS / "time_oos_ci.csv")
    lo, hi = ci["pooled_mnl"]["sharpe_diff"]
    log(f"pooled_mnl sharpe diff vs equal_weight: {ci['pooled_mnl']['sharpe_diff']}")
    if lo > 0 and hi > 0:
        verdict = "H2 SUPPORTED"
    elif hi < 0:
        verdict = "H2 FAILED (significantly worse)"
    else:
        verdict = "H2 NOT SUPPORTED (CI includes 0)"
    log(verdict)

    # Robustness: same bootstrap with the Politis-White automatic block length.
    ci_pw = {
        name: paired_block_bootstrap(
            daily[name], daily["equal_weight"],
            base.bootstrap.n_resamples, base.bootstrap.block_length,
            base.bootstrap.confidence, np.random.default_rng(seed_for(name + "_pw", base.seed)), True,
        )
        for name in ("per_stock", "pooled_mnl")
    }
    pd.DataFrame(ci_pw).T.to_csv(REPORTS / "time_oos_ci_pw.csv")
    log(f"PW block length={politis_white_block_length(daily['pooled_mnl'].to_numpy())} "
        f"-> pooled_mnl sharpe diff {ci_pw['pooled_mnl']['sharpe_diff']}")
    save_series(daily, REPORTS / "time_oos_daily.csv")
    write_run_meta(REPORTS / "time_oos_meta.json", cfg, {"period": f"{START}..{END}", "rebalances": len(windows)})
    log("wrote time_oos_eval.csv, time_oos_ci.csv, time_oos_ci_pw.csv, time_oos_daily.csv, time_oos_meta.json")


if __name__ == "__main__":
    main()
