"""Two-stage portfolio construction and baseline allocators.

Two-stage (the engine):
    1. allocate **across clusters** (equal / risk-parity / light forecast tilt);
    2. allocate **within each cluster** (equal / inverse-volatility);
    under a **per-stock cap** ``cap = max(cap_floor, cap_factor / N)``.

Cap feasibility: cluster weights are clipped at ``size x cap`` and the excess is
redistributed iteratively to clusters with remaining room; the same is done within a
cluster at the stock level. This guarantees weights sum to 1, are non-negative, respect
the cap, and stay feasible for small universes (the floor alone is not: 8 x 10% = 80%).

Baselines (comparisons, not the engine):
    equal-weight; **minimum-variance** (main) and mean-variance with shrunk means, both
    long-only, capped, Ledoit-Wolf shrinkage; correlation-clustering; random clusters of
    the same sizes; sector clusters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from genomefolio.config import ClusterConfig, PortfolioConfig


def cap_redistribute(weights, cap, max_iter: int = 200, tol: float = 1e-12) -> np.ndarray:
    """Clip entries above their cap and redistribute the excess to entries with room.

    ``cap`` may be a scalar or a per-entry array. Preserves the total. Raises
    ``ValueError`` when the caps are infeasible for the total (``sum(cap) < total``).
    """
    w = np.array(weights, dtype=float)
    c = np.broadcast_to(np.asarray(cap, dtype=float), w.shape).astype(float).copy()
    total = float(w.sum())
    if total <= 0.0:
        return w
    if c.sum() < total - tol:
        raise ValueError("infeasible cap: sum(cap) < total weight")
    for _ in range(max_iter):
        excess = float(np.maximum(w - c, 0.0).sum())
        if excess <= tol:
            break
        w = np.minimum(w, c)
        free = w < c - tol
        if not free.any():
            break
        room = (c - w)[free]
        w[free] += excess * room / room.sum()
    return w


def _inverse_vol(returns_window: pd.DataFrame) -> np.ndarray:
    vol = returns_window.std(ddof=0).to_numpy()
    vol = np.where((vol > 0) & np.isfinite(vol), vol, np.nan)
    if np.all(np.isnan(vol)):
        inv = np.ones_like(vol)
    else:
        inv = 1.0 / np.where(np.isnan(vol), np.nanmean(vol), vol)
    return inv / inv.sum()


def _ledoit_wolf_vols(frame: pd.DataFrame) -> np.ndarray:
    """Volatilities from a Ledoit-Wolf shrunk covariance (robust to estimation error)."""
    from sklearn.covariance import LedoitWolf

    X = frame.to_numpy(dtype=float)
    if X.shape[1] == 1:
        return np.array([float(np.std(X[:, 0], ddof=0))])
    cov = LedoitWolf().fit(X).covariance_
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def _cluster_first_stage(
    clusters: dict[int, tuple[str, ...]],
    returns_window: pd.DataFrame,
    cfg: PortfolioConfig,
    forecasts: dict[int, float] | None,
) -> np.ndarray:
    ids = sorted(clusters)
    sizes = np.array([len(clusters[c]) for c in ids], dtype=float)

    if cfg.cluster_method in {"risk_parity", "risk_parity_lw"}:
        composites = pd.DataFrame(
            {c: returns_window[list(clusters[c])].mean(axis=1) for c in ids}
        )
        if cfg.cluster_method == "risk_parity_lw" and composites.shape[1] > 1:
            vols = _ledoit_wolf_vols(composites)
        else:
            vols = composites.std(ddof=0).to_numpy()
        inv = 1.0 / np.where(vols > 0, vols, np.nanmean(vols))
        w = inv / inv.sum()
    elif cfg.cluster_method == "forecast_tilt" and forecasts is not None:
        f = np.array([forecasts[c] for c in ids], dtype=float)
        if f.size > 1 and f.std() > 0:
            z = (f - f.mean()) / f.std()
        else:
            z = np.zeros_like(f)
        tilted = np.clip(1.0 + cfg.forecast_tilt_strength * z, 0.05, None)
        w = tilted / tilted.sum()
    else:  # equal (also the fallback when forecast_tilt has no forecasts)
        w = np.full(len(ids), 1.0 / len(ids))

    caps = sizes * cfg.cap(sum(len(m) for m in clusters.values()))
    return cap_redistribute(w, cap=caps)


def two_stage_weights(
    clusters: dict[int, tuple[str, ...]],
    returns_window: pd.DataFrame,
    cfg: PortfolioConfig,
    forecasts: dict[int, float] | None = None,
) -> pd.Series:
    """Return portfolio weights across all stocks for one rebalance."""
    tickers = [t for c in sorted(clusters) for t in clusters[c]]
    if not tickers:
        raise ValueError("two_stage_weights: clusters are empty (no stocks to allocate)")
    n = len(tickers)
    cap = cfg.cap(n)
    cluster_w = _cluster_first_stage(clusters, returns_window, cfg, forecasts)

    weights: dict[str, float] = {}
    for idx, c in enumerate(sorted(clusters)):
        members = list(clusters[c])
        sub = returns_window[members]
        if cfg.within_cluster == "equal":
            base = np.full(len(members), 1.0 / len(members))
        elif cfg.within_cluster == "inverse_vol_lw":
            vols = _ledoit_wolf_vols(sub)
            inv = 1.0 / np.where(vols > 0, vols, np.nanmean(vols) if np.any(vols > 0) else 1.0)
            base = inv / inv.sum()
        else:
            base = _inverse_vol(sub)
        block = base * cluster_w[idx]
        block = cap_redistribute(block, cap=cap)
        for t, w in zip(members, block):
            weights[t] = float(w)
    out = pd.Series(weights).reindex(tickers)
    out = out / out.sum()
    return out


def equal_weights(tickers) -> pd.Series:
    """Equal-weight across a single group (the gate-failure fallback)."""
    tickers = list(tickers)
    if not tickers:
        raise ValueError("equal_weights: no tickers given")
    return pd.Series(1.0 / len(tickers), index=tickers)


def optimised_weights(
    returns_window: pd.DataFrame,
    cap: float,
    risk_aversion: float = 0.0,
    mean_shrink: float = 0.0,
) -> pd.Series:
    """Long-only, capped, sum-to-1 optimiser with Ledoit-Wolf covariance.

    ``risk_aversion == 0`` gives the **minimum-variance** portfolio; ``> 0`` gives the
    mean-variance variant, with ``mean_shrink`` shrinking historical means toward zero
    (``0`` = no shrinkage) to tame estimation error.
    """
    from sklearn.covariance import LedoitWolf

    tickers = list(returns_window.columns)
    n = len(tickers)
    X = returns_window.to_numpy(dtype=float)
    cov = LedoitWolf().fit(X).covariance_
    mu = X.mean(axis=0)
    if mean_shrink > 0.0:
        mu = mu * (1.0 - mean_shrink)

    def objective(w):
        return float(w @ cov @ w - risk_aversion * (w @ mu))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, cap)] * n
    res = minimize(
        objective, np.full(n, 1.0 / n), method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, cap)
    if not np.isclose(w.sum(), 1.0) or not np.isfinite(w).all():
        w = cap_redistribute(np.full(n, 1.0 / n), cap=cap)
    w = w / w.sum()
    return pd.Series(w, index=tickers)


def correlation_clusters(returns_window: pd.DataFrame, cluster_cfg: ClusterConfig) -> dict[int, tuple[str, ...]]:
    """Cluster stocks by correlation distance (``1 - corr``) with the same k rules."""
    from genomefolio.cluster import select_constrained_k

    corr = returns_window.corr().to_numpy()
    D = 1.0 - corr
    np.fill_diagonal(D, 0.0)
    frame = pd.DataFrame(D, index=returns_window.columns, columns=returns_window.columns)
    return select_constrained_k(frame, cluster_cfg).clusters


def random_clusters(sizes, tickers, rng: np.random.Generator) -> dict[int, tuple[str, ...]]:
    """Random partition with the same cluster sizes as the alignment clusters."""
    pool = np.array(list(tickers), dtype=object)
    rng.shuffle(pool)
    out: dict[int, tuple[str, ...]] = {}
    start = 0
    for i, size in enumerate(sizes):
        out[i] = tuple(pool[start : start + size].tolist())
        start += size
    return out


def sector_clusters(
    sector_map: dict[str, tuple[str, ...]], tickers
) -> dict[int, tuple[str, ...]]:
    """Sector groups restricted to the traded universe."""
    universe = set(tickers)
    out: dict[int, tuple[str, ...]] = {}
    for i, (_, members) in enumerate(sorted(sector_map.items())):
        group = tuple(t for t in members if t in universe)
        if group:
            out[i] = group
    return out
