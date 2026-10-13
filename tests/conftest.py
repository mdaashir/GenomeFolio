"""Shared test fixtures: toy symbol frames and known block-structure distance matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def block_distance():
    """Factory for a two-block distance matrix (within-block small, between large)."""

    def make(n: int = 12, within: float = 0.1, between: float = 0.9) -> pd.DataFrame:
        labels = np.array([0] * (n // 2) + [1] * (n // 2))
        D = np.where(labels[:, None] == labels[None, :], within, between)
        np.fill_diagonal(D, 0.0)
        cols = [f"S{i}" for i in range(n)]
        return pd.DataFrame(D, index=cols, columns=cols)

    return make


@pytest.fixture
def symbol_frame():
    """Factory wrapping a (length x n_stocks) array as an integer symbol frame."""

    def make(data, columns=None) -> pd.DataFrame:
        data = np.asarray(data)
        cols = columns or [f"S{i}" for i in range(data.shape[1])]
        idx = pd.date_range("2020-01-01", periods=data.shape[0], freq="B")
        return pd.DataFrame(data, index=idx, columns=cols)

    return make


@pytest.fixture
def separable_symbols(symbol_frame):
    """Factory for a symbol frame with two groups of identical strings (clear blocks)."""

    def make(n_per_group: int = 5, length: int = 80, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        base_a = rng.integers(0, 5, size=length)
        base_b = rng.integers(0, 5, size=length)
        data = np.column_stack(
            [base_a for _ in range(n_per_group)] + [base_b for _ in range(n_per_group)]
        )
        return symbol_frame(data)

    return make
