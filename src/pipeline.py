"""Clustering followed directly by forecasting.

The shortened pipeline: data -> symbolize -> align -> cluster -> forecast. There is no
null gate and no portfolio/backtest downstream: after the constrained-k clustering, each
cluster's equal-weight composite return series is forecast over the coming horizon.

Inputs: a returns window (dates x tickers) and a :class:`~src.config.Config`.
Outputs: a :class:`PipelineResult` with cluster membership and per-cluster forecasts.

No look-ahead: the caller passes the training window only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.align import distance_matrix
from src.cluster import select_constrained_k
from src.config import Config
from src.forecast import ClusterForecast, forecast_clusters
from src.symbolize import symbolize


@dataclass
class PipelineResult:
    """Clusters and their forecasts for one window."""

    clusters: dict[int, tuple[str, ...]]
    forecasts: dict[int, ClusterForecast]
    n_clusters: int
    silhouette: float
    min_size_ok: bool


def run_pipeline(returns_window: pd.DataFrame, cfg: Config) -> PipelineResult:
    """Symbolize, align, cluster, then forecast each cluster composite."""
    symbols = symbolize(
        returns_window,
        cfg.symbolize.cutoffs,
        market_neutral=cfg.symbolize.market_neutral,
        quantile_scope=cfg.symbolize.quantile_scope,
    )
    D = distance_matrix(symbols, cfg.align)
    clustering = select_constrained_k(D, cfg.cluster, cfg.cluster.linkage)

    clusters = clustering.clusters
    composite = pd.DataFrame(
        {cid: returns_window[list(members)].mean(axis=1) for cid, members in clusters.items()}
    )
    forecasts = forecast_clusters(composite, cfg.forecast, cfg.window.forecast_horizon)
    return PipelineResult(
        clusters=clusters,
        forecasts=forecasts,
        n_clusters=clustering.n_clusters,
        silhouette=clustering.silhouette,
        min_size_ok=clustering.min_size_ok,
    )
