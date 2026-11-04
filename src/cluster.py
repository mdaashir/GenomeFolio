"""Clustering of the alignment distance matrix.

Hierarchical average linkage (UPGMA) on the normalised alignment distance, with a
**constrained k selection**: silhouette maximised over ``k`` in
``[k_min, min(k_max, floor(N / divisor))]`` subject to **every cluster having at least
``min_cluster_size`` members** (silhouette otherwise favours k = 2 with ~20 stocks).

``select_constrained_k`` is the single source of truth for the k-selection rule.

Also provides a **band sweep** (re-cluster at several band widths) plus Adjusted Rand
Index stability, so the chosen solution is not an artefact of one band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.config import AlignConfig, ClusterConfig


@dataclass
class ClusteringResult:
    """A single clustering solution."""

    labels: pd.Series  # index = stock, values = 0-based cluster ids
    n_clusters: int
    silhouette: float
    min_size_ok: bool

    @property
    def clusters(self) -> dict[int, tuple[str, ...]]:
        out: dict[int, list[str]] = {}
        for stock, lab in self.labels.items():
            out.setdefault(int(lab), []).append(stock)
        return {k: tuple(v) for k, v in sorted(out.items())}


def candidate_partitions(
    D: pd.DataFrame, cfg: ClusterConfig, linkage_method: str = "average"
) -> list[dict]:
    """Evaluate every k in the valid range.

    Returns one dict per ``k`` with ``k``, ``silhouette``, ``min_size_ok``, ``sizes`` and the
    raw ``labels``. Shared by the selection rule and the UI/plots so both use the same
    candidate set.
    """
    stocks = list(D.index)
    n = len(stocks)
    if n < 2:
        raise ValueError("need at least two stocks to cluster")
    condensed = squareform(D.to_numpy(), checks=False)
    Z = linkage(condensed, method=linkage_method)
    cap = cfg.k_cap(n)
    out: list[dict] = []
    for k in range(cfg.k_min, cap + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        sizes = np.bincount(labels)[1:]
        ok = bool(sizes.size and sizes.min() >= cfg.min_cluster_size)
        sil = float(silhouette_score(D.to_numpy(), labels, metric="precomputed"))
        out.append(
            {"k": k, "silhouette": sil, "min_size_ok": ok, "sizes": sizes.tolist(), "labels": labels}
        )
    return out


def silhouette_by_k(
    D: pd.DataFrame, cfg: ClusterConfig, linkage_method: str = "average"
) -> dict[int, float]:
    """Silhouette score for each candidate k (for the UI and plots)."""
    return {c["k"]: c["silhouette"] for c in candidate_partitions(D, cfg, linkage_method)}


def select_constrained_k(
    D: pd.DataFrame, cfg: ClusterConfig, linkage_method: str = "average"
) -> ClusteringResult:
    """Constrain the k search and pick the best-silhouette partition.

    ``D`` must be a square, symmetric, non-negative distance frame. Returns a
    :class:`ClusteringResult`; ``min_size_ok`` is False only if no k satisfied the
    minimum-size rule, in which case the best-silhouette (possibly undersized) cut is
    returned rather than failing.
    """
    stocks = list(D.index)
    n = len(stocks)
    if n < 2:
        raise ValueError("need at least two stocks to cluster")
    if cfg.k_cap(n) < cfg.k_min:
        # Too few stocks for the size rule to allow multiple clusters.
        return ClusteringResult(
            labels=pd.Series(np.zeros(n, dtype=int), index=stocks),
            n_clusters=1,
            silhouette=float("nan"),
            min_size_ok=False,
        )

    candidates = candidate_partitions(D, cfg, linkage_method)
    ok_candidates = [c for c in candidates if c["min_size_ok"]]
    pool = ok_candidates if ok_candidates else candidates
    best = max(pool, key=lambda c: float(c["silhouette"]))
    return ClusteringResult(
        labels=pd.Series(best["labels"] - 1, index=stocks),
        n_clusters=int(best["k"]),
        silhouette=float(best["silhouette"]),
        min_size_ok=bool(best["min_size_ok"]),
    )


def band_sweep(
    symbols: pd.DataFrame, align_cfg: AlignConfig, cluster_cfg: ClusterConfig, bands
) -> dict[int, ClusteringResult]:
    """Cluster at each band width in ``bands`` (re-aligning each time)."""
    from src.align import distance_matrix

    out: dict[int, ClusteringResult] = {}
    for w in bands:
        D = distance_matrix(symbols, align_cfg, band=int(w))
        out[int(w)] = select_constrained_k(D, cluster_cfg, cluster_cfg.linkage)
    return out


def stability_ari(results: dict[int, ClusteringResult]) -> pd.DataFrame:
    """Pairwise Adjusted Rand Index between the band-sweep clusterings."""
    bands = sorted(results)
    M = np.ones((len(bands), len(bands)))
    for i, bi in enumerate(bands):
        for j, bj in enumerate(bands):
            if i == j:
                continue
            li = results[bi].labels.reindex(results[bj].labels.index).to_numpy()
            lj = results[bj].labels.to_numpy()
            M[i, j] = adjusted_rand_score(li, lj)
    return pd.DataFrame(M, index=bands, columns=bands)
