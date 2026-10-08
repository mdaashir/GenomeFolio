"""Rolling-update schedule.

Owns the walk-forward windowing so the engine (``src/backtest.py``) stays a thin loop:
at each rebalance date we re-symbolize, re-align, re-cluster and rebalance, using only
the trailing training window (no look-ahead).

``rebalance_positions`` returns the integer offsets into the returns frame at which a
rebalance occurs; ``walk_forward_windows`` yields ``(position, rebalance_date, window,
forward)`` for each of those offsets, where ``window`` is the training window ending at
the rebalance and ``forward`` is the holding period that follows.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd


def rebalance_positions(n: int, train_days: int, rebalance_days: int) -> list[int]:
    """Integer offsets for each rebalance: start at ``train_days``, step ``rebalance_days``."""
    positions = []
    i = train_days
    while i + rebalance_days <= n:
        positions.append(i)
        i += rebalance_days
    return positions


def walk_forward_windows(
    returns: pd.DataFrame, train_days: int, rebalance_days: int
) -> Iterator[tuple[int, pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(position, date, training_window, forward_window)`` for every rebalance."""
    for i in rebalance_positions(len(returns), train_days, rebalance_days):
        yield i, returns.index[i], returns.iloc[i - train_days : i], returns.iloc[i : i + rebalance_days]
