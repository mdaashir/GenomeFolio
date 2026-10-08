"""Symbolization ablations (+ magnitude-aware alignment).

Tests whether the symbolization choice — which discards magnitude — is what kills the
alignment signal. Variants:

- ``per_stock 20/40/60/80`` (baseline),
- ``per_stock 15/35/65/85``,
- ``per_stock`` + market-neutral returns,
- ``pooled 20/40/60/80`` (**magnitude-aware**: thresholds from all stocks pooled, so
  relative volatility survives),
- ``pooled 15/35/65/85``,
- ``pooled`` + market-neutral,
- correlation-distance clustering, for reference.

For speed and because the null gate passes ~90% of rebalances, the gate is **bypassed**
here: clusters are used directly. Every variant is treated identically, so the comparison
between symbolizations is fair. All variants use the best-performing allocator from the
follow-up experiments (risk-parity across clusters, inverse-vol within).

Usage::

    uv run python scripts/run_symbolize_ablations.py
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.align import distance_matrix
from src.backtest import paired_block_bootstrap, performance_metrics
from src.cluster import select_constrained_k
from src.config import load_config
from src.data import align_to_common_days, get_prices, time_split, to_returns
from src.portfolio import correlation_clusters, equal_weights, two_stage_weights
from src.rolling import walk_forward_windows
from src.symbolize import symbolize

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
LOG = REPORTS / "ablations.log"
COST_BPS = 10.0
BASE_CUT = (0.20, 0.40, 0.60, 0.80)
ALT_CUT = (0.15, 0.35, 0.65, 0.85)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


VARIANTS = [
    ("per_stock_20_40_60_80", BASE_CUT, False, "per_stock"),
    ("per_stock_15_35_65_85", ALT_CUT, False, "per_stock"),
    ("per_stock_20_40_60_80_mnl", BASE_CUT, True, "per_stock"),
    ("pooled_20_40_60_80", BASE_CUT, False, "pooled"),
    ("pooled_15_35_65_85", ALT_CUT, False, "pooled"),
    ("pooled_20_40_60_80_mnl", BASE_CUT, True, "pooled"),
]


def _portfolio(daily_parts, tickers):
    daily = pd.concat(daily_parts).sort_index()
    return daily


def main(period: str = "report") -> None:
    LOG = REPORTS / f"ablations_{period}.log"
    LOG.write_text("", encoding="utf-8")
    cfg = load_config()
    returns = align_to_common_days(to_returns(get_prices(cfg)))
    tune, report = time_split(returns, cfg.split.tune_fraction, enabled=cfg.split.enabled)
    if period == "tune":
        target = tune
    elif period == "all":
        target = returns
    else:
        target = report if len(report) else returns
    tickers = list(target.columns)
    pcfg = replace(cfg.portfolio, cluster_method="risk_parity", within_cluster="inverse_vol")
    windows = list(walk_forward_windows(target, cfg.window.train_days, cfg.window.rebalance_days))
    log(f"period={period} days={len(target)} tickers={len(tickers)} rebalances={len(windows)}")

    daily: dict[str, pd.Series] = {}
    tables: dict[str, dict] = {}
    labels_by_variant: dict[str, list[pd.Series]] = {}

    def evaluate(name, parts, labels, ks):
        d = _portfolio(parts, tickers)
        daily[name] = d
        m = performance_metrics(d)
        m["avg_k"] = float(np.mean(ks)) if ks else np.nan
        tables[name] = m
        labels_by_variant[name] = labels
        log(f"{name}: sharpe={m['sharpe']:.3f} ann_return={m['ann_return']:.4f} avg_k={m['avg_k']:.2f}")

    for name, cutoffs, mn, scope in VARIANTS:
        parts, labels, ks, prev = [], [], [], pd.Series(0.0, index=tickers)
        for _, _, window, forward in windows:
            symbols = symbolize(window, cutoffs, market_neutral=mn, quantile_scope=scope)
            D = distance_matrix(symbols, cfg.align)
            res = select_constrained_k(D, cfg.cluster, cfg.cluster.linkage)
            w = two_stage_weights(res.clusters, window, pcfg).reindex(tickers).fillna(0.0)
            turn = float((w - prev).abs().sum())
            prev = w
            seg = (forward * w).sum(axis=1).copy()
            seg.iloc[0] -= (COST_BPS / 1e4) * turn
            parts.append(seg)
            labels.append(res.labels.reindex(tickers))
            ks.append(res.n_clusters)
        evaluate(name, parts, labels, ks)

    # correlation-distance reference with the same allocator
    parts, labels, ks, prev = [], [], [], pd.Series(0.0, index=tickers)
    for _, _, window, forward in windows:
        clusters = correlation_clusters(window, cfg.cluster)
        w = two_stage_weights(clusters, window, pcfg).reindex(tickers).fillna(0.0)
        turn = float((w - prev).abs().sum())
        prev = w
        seg = (forward * w).sum(axis=1).copy()
        seg.iloc[0] -= (COST_BPS / 1e4) * turn
        parts.append(seg)
        labels.append(pd.Series({t: i for i, c in clusters.items() for t in c}).reindex(tickers))
        ks.append(len(clusters))
    evaluate("correlation_ref", parts, labels, ks)

    # equal-weight reference
    parts, prev = [], pd.Series(0.0, index=tickers)
    for _, _, _, forward in windows:
        w = equal_weights(tickers)
        turn = float((w - prev).abs().sum())
        prev = w
        seg = (forward * w).sum(axis=1).copy()
        seg.iloc[0] -= (COST_BPS / 1e4) * turn
        parts.append(seg)
    evaluate("equal_weight", parts, [], [])

    pd.DataFrame(tables).T.to_csv(REPORTS / f"ablations_{period}.csv")

    # how much do the clusters move relative to the baseline symbolization?
    base = labels_by_variant["per_stock_20_40_60_80"]
    stability = {}
    for name, labels in labels_by_variant.items():
        if not labels:
            continue
        stability[name] = float(
            np.mean([adjusted_rand_score(b.to_numpy(), l.to_numpy()) for b, l in zip(base, labels)])
        )
    pd.Series(stability, name="mean_ari_vs_baseline").to_csv(REPORTS / f"ablations_stability_{period}.csv")
    log(f"mean ARI vs baseline: {stability}")

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
    pd.DataFrame(ci).T.to_csv(REPORTS / f"ablations_ci_{period}.csv")
    log(f"wrote ablations_{period}.csv, ablations_ci_{period}.csv, ablations_stability_{period}.csv")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "report")
