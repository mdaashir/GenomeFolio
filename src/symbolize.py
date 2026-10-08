"""Symbolization: turn daily returns into a 5-symbol sequence.

Alphabet (ordinal position matters for the graded substitution matrix):
    0 = big-down, 1 = small-down, 2 = flat, 3 = small-up, 4 = big-up

Inputs: a returns frame (dates x tickers) and per-stock quantile cutoffs, e.g.
``(0.20, 0.40, 0.60, 0.80)``.

Outputs: an integer-coded frame of the same shape, values in ``{0, 1, 2, 3, 4}``.

Assumptions / no look-ahead:
- The cutoffs are computed from the *values passed in*, so callers must pass only the
  current training window. Nothing here reaches beyond the given slice.
- Cutoffs are per-stock, so each stock's own volatility is normalised away: the strings
  encode *when* moves happen (shape/timing), not *how big* they are.
- A constant series (all values equal) is emitted as all ``flat`` — there is no
  ordering to encode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BIG_DOWN, SMALL_DOWN, FLAT, SMALL_UP, BIG_UP = 0, 1, 2, 3, 4
N_SYMBOLS = 5
SYMBOL_NAMES = {BIG_DOWN: "D2", SMALL_DOWN: "D1", FLAT: "F", SMALL_UP: "U1", BIG_UP: "U2"}


def quantile_edges(values: np.ndarray, cutoffs: tuple[float, ...]) -> np.ndarray:
    """Return the quantile edges for one series (NaNs ignored).

    Raises ``ValueError`` on a series that is too short or entirely NaN.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[~np.isnan(arr)]
    if finite.size < 2:
        raise ValueError("need at least two non-NaN observations to compute quantiles")
    return np.quantile(finite, cutoffs)


def pooled_quantile_edges(returns: pd.DataFrame, cutoffs: tuple[float, ...]) -> np.ndarray:
    """Quantile edges from all stocks' returns pooled together.

    Unlike per-stock edges, these preserve relative volatility: a high-vol stock draws
    more extreme symbols than a low-vol one, so the strings encode magnitude as well as
    timing (the "alignment with magnitude" ablation).
    """
    return quantile_edges(returns.to_numpy().ravel(), cutoffs)


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map values to 0..4 by the edges; constant series collapse to ``flat``."""
    arr = np.asarray(values, dtype=float)
    if np.all(edges == edges[0]):
        codes = np.full(arr.shape, FLAT, dtype=float)
    else:
        codes = np.digitize(arr, edges, right=False).astype(float)
    codes[np.isnan(arr)] = np.nan
    return codes


def market_neutralize(returns: pd.DataFrame) -> pd.DataFrame:
    """Stock return minus the equal-weight universe return, row by row."""
    return returns.sub(returns.mean(axis=1), axis=0)


def symbolize_series(values, cutoffs) -> pd.Series:
    """Symbolize a single returns series using its own quantile cutoffs."""
    edges = quantile_edges(np.asarray(values, dtype=float), cutoffs)
    return pd.Series(_digitize(np.asarray(values, dtype=float), edges), index=values.index)


def symbolize(
    returns: pd.DataFrame,
    cutoffs: tuple[float, ...],
    market_neutral: bool = False,
    quantile_scope: str = "per_stock",
) -> pd.DataFrame:
    """Symbolize every column of ``returns`` into an integer-coded frame.

    Parameters
    - ``market_neutral``: de-mean by the equal-weight universe per day first.
    - ``quantile_scope``: ``"per_stock"`` (default) computes each stock's own edges, so
      magnitude is normalised away and strings encode timing/shape; ``"pooled"`` computes
      edges from all stocks pooled, preserving relative volatility (magnitude-aware).
    """
    if quantile_scope not in {"per_stock", "pooled"}:
        raise ValueError(f"quantile_scope must be per_stock|pooled, got {quantile_scope!r}")
    data = market_neutralize(returns) if market_neutral else returns
    if quantile_scope == "pooled":
        edges = pooled_quantile_edges(data, cutoffs)
        out = {
            col: pd.Series(_digitize(data[col].to_numpy(dtype=float), edges), index=data.index)
            for col in data.columns
        }
        return pd.DataFrame(out, index=data.index)
    out = {col: symbolize_series(data[col], cutoffs) for col in data.columns}
    return pd.DataFrame(out, index=data.index)
