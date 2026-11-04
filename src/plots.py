"""Matplotlib figures for each pipeline phase (headless; used by the Streamlit UI)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram as _dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from src.cluster import band_sweep, silhouette_by_k, stability_ari
from src.config import AlignConfig, ClusterConfig

SYMBOL_NAMES = ["D2", "D1", "F", "U1", "U2"]
_SYMBOL_COLOURS = ["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"]


def _annotate(ax, data: np.ndarray, fmt: str) -> None:
    n = data.shape[0]
    if n > 20:
        return
    for i in range(n):
        for j in range(n):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=6.5)


def cumulative_returns(returns: pd.DataFrame, title: str = "Cumulative returns (growth of 1)") -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    growth = (1.0 + returns).cumprod()
    for col in growth.columns:
        ax.plot(growth.index, growth[col].to_numpy(), lw=1.1, label=str(col))
    ax.set_title(title)
    ax.set_ylabel("Growth of 1")
    ax.grid(alpha=0.3)
    if growth.shape[1] <= 12:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def symbol_heatmap(symbols: pd.DataFrame, title: str = "Symbolised returns") -> Figure:
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.34 * len(symbols.columns))))
    data = np.ma.masked_invalid(symbols.to_numpy().T.astype(float))
    cmap = ListedColormap(_SYMBOL_COLOURS).with_extremes(bad="#d9d9d9")
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_yticks(range(len(symbols.columns)))
    ax.set_yticklabels(symbols.columns, fontsize=8)
    n = symbols.shape[0]
    ticks = np.linspace(0, n - 1, min(6, n)).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.Timestamp(symbols.index[i]).strftime("%b %Y") for i in ticks], fontsize=8)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(SYMBOL_NAMES)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def symbol_distribution(symbols: pd.DataFrame, title: str = "Symbol frequency per stock") -> Figure:
    shares = (
        symbols.apply(lambda c: c.value_counts(normalize=True))
        .T.reindex(columns=range(5), fill_value=0.0)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(shares))
    for code in range(5):
        values = shares[code].to_numpy()
        ax.bar(
            shares.index,
            values,
            bottom=bottom,
            color=_SYMBOL_COLOURS[code],
            edgecolor="grey",
            linewidth=0.3,
            label=SYMBOL_NAMES[code],
        )
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.set_title(title)
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return fig


def _matrix_heatmap(matrix: pd.DataFrame, title: str, fmt: str, cmap: str) -> Figure:
    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.45 * n), max(4.0, 0.45 * n)))
    im = ax.imshow(matrix.to_numpy(), cmap=cmap)
    ax.set_xticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=60, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(matrix.index, fontsize=7)
    _annotate(ax, matrix.to_numpy(), fmt)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def score_heatmap(scores: pd.DataFrame) -> Figure:
    return _matrix_heatmap(scores, "Raw alignment scores", "{:.1f}", "viridis")


def distance_heatmap(distance: pd.DataFrame) -> Figure:
    return _matrix_heatmap(
        distance, "Pairwise distance (1 - normalised score)", "{:.2f}", "magma"
    )


def dendrogram(distance: pd.DataFrame, linkage_method: str = "average") -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    Z = linkage(squareform(distance.to_numpy(), checks=False), method=linkage_method)
    _dendrogram(Z, labels=list(distance.index), ax=ax, leaf_rotation=60)
    ax.set_title("Hierarchical clustering dendrogram")
    fig.tight_layout()
    return fig


def silhouette_curve(distance: pd.DataFrame, cluster_cfg: ClusterConfig) -> Figure:
    scores = silhouette_by_k(distance, cluster_cfg)
    ks = list(scores)
    values = [scores[k] for k in ks]
    best_k = max(scores, key=scores.get) if scores else None
    colours = ["#2166ac" if k == best_k else "#a6bddb" for k in ks]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar([str(k) for k in ks], values, color=colours)
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette")
    ax.set_title("Silhouette by candidate k (best highlighted)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def cluster_sizes(clusters: Mapping[int, Sequence[str]]) -> Figure:
    ids = sorted(clusters)
    sizes = [len(clusters[c]) for c in ids]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar([f"cluster {c}" for c in ids], sizes, color="#2166ac")
    ax.set_ylabel("stocks")
    ax.set_title("Cluster sizes")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def forecast_panels(
    composite_returns: pd.DataFrame, forecasts: Mapping, horizon: int
) -> Figure:
    ids = sorted(forecasts)
    fig, axes = plt.subplots(len(ids), 1, figsize=(10, max(2.6, 2.8 * len(ids))), squeeze=False)
    for ax, cid in zip(axes[:, 0], ids):
        history = composite_returns[cid].cumsum()
        ax.plot(history.index, history.to_numpy(), lw=1.2, label="history")
        path = forecasts[cid].path
        if path:
            future = pd.bdate_range(history.index[-1], periods=len(path) + 1)[1:]
            ax.plot(
                future,
                history.iloc[-1] + np.asarray(path),
                lw=1.4,
                ls="--",
                color="#b2182b",
                label="forecast",
            )
        forecast = forecasts[cid]
        ax.set_title(
            f"Cluster {cid} — forecast {forecast.expected_return:+.4f} over {horizon}d "
            f"[{forecast.model}]",
            fontsize=9,
        )
        ax.set_ylabel("cumulative return")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def band_stability(
    symbols: pd.DataFrame, align_cfg: AlignConfig, cluster_cfg: ClusterConfig, bands
) -> tuple[Figure, pd.DataFrame, pd.DataFrame]:
    sweep = band_sweep(symbols, align_cfg, cluster_cfg, bands)
    ari = stability_ari(sweep)
    summary = pd.DataFrame(
        {
            "k": [r.n_clusters for r in sweep.values()],
            "silhouette": [r.silhouette for r in sweep.values()],
        },
        index=list(sweep),
    )
    summary.index.name = "band"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.bar([str(b) for b in summary.index], summary["k"], color="#2166ac")
    ax1.set_title("Clusters per band")
    ax1.set_xlabel("band w")
    ax1.set_ylabel("k")
    ax1.grid(axis="y", alpha=0.3)
    values = ari.to_numpy()
    im = ax2.imshow(values, cmap="viridis", vmin=0, vmax=1)
    ax2.set_xticks(range(len(ari.columns)))
    ax2.set_xticklabels(ari.columns)
    ax2.set_yticks(range(len(ari.index)))
    ax2.set_yticklabels(ari.index)
    ax2.set_title("Adjusted Rand Index across bands")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            colour = "white" if values[i, j] < 0.5 else "black"
            ax2.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=colour)
    fig.colorbar(im, ax=ax2)
    fig.tight_layout()
    return fig, summary, ari
