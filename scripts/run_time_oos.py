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
import run_holdout_eval as h  # noqa: E402  (shared arm builder)

from src.backtest import paired_block_bootstrap, performance_metrics  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import align_to_common_days, get_prices, to_returns  # noqa: E402
from src.portfolio import equal_weights  # noqa: E402
from src.rolling import walk_forward_windows  # noqa: E402

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

    rng = np.random.default_rng(base.seed + 999)
    ci = {
        name: paired_block_bootstrap(
            daily[name], daily["equal_weight"],
            base.bootstrap.n_resamples, base.bootstrap.block_length,
            base.bootstrap.confidence, rng, base.bootstrap.use_politis_white,
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
    log("wrote time_oos_eval.csv, time_oos_ci.csv")


if __name__ == "__main__":
    main()
