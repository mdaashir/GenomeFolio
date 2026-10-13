"""Plotting helpers (matplotlib, headless).

Writes figures to a directory (default ``reports/figures``, gitignored). Kept minimal and
dependency-light so results can be visualised from the persisted daily series.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def cumulative_returns_plot(
    daily_by_name: Mapping[str, pd.Series], path: str | Path, title: str = "Cumulative returns"
) -> Path:
    """Plot cumulative growth of several daily return series to ``path`` (PNG)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, series in daily_by_name.items():
        growth = (1.0 + series.dropna()).cumprod()
        ax.plot(growth.index, growth.to_numpy(), label=name, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Growth of 1")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def turnover_bar_plot(turnover: pd.Series, path: str | Path, title: str = "Turnover per rebalance") -> Path:
    """Bar chart of per-rebalance turnover to ``path`` (PNG)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(turnover)), turnover.to_numpy())
    ax.set_title(title)
    ax.set_ylabel("sum |dw|")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
