"""Dynamic time warping on raw returns — a non-alignment pattern-similarity baseline.

Where sequence alignment discretises returns into symbols, DTW works on the raw return
values and matches them with a Sakoe-Chiba band. It is a fair control for "does
symbolisation help at all?". Equal-length training windows keep it O(n * w).

The returned distance is the optimal warp-path cost normalised by the two path lengths
(``n + m``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn

        return wrap

_INF = 1.0e18


@njit(cache=True)
def _dtw_cost(a, b, band):  # pragma: no cover
    n = a.shape[0]
    m = b.shape[0]
    D = np.full((n + 1, m + 1), _INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = i - band
        if lo < 1:
            lo = 1
        hi = i + band
        if hi > m:
            hi = m
        for j in range(lo, hi + 1):
            c = abs(a[i - 1] - b[j - 1])
            best = D[i - 1, j - 1]
            if D[i - 1, j] < best:
                best = D[i - 1, j]
            if D[i, j - 1] < best:
                best = D[i, j - 1]
            D[i, j] = c + best
    return D[n, m]


def dtw_distance(a, b, band: int | None = None) -> float:
    """Normalised DTW distance between two equal-or-unequal length series."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    if band is None:
        band = max(n, m)
    if band < abs(n - m):
        raise ValueError(f"band ({band}) < |len(a)-len(b)| ({abs(n - m)}): end cell unreachable")
    return float(_dtw_cost(a, b, band) / (n + m))


def dtw_distance_matrix(returns_window: pd.DataFrame, band: int = 10) -> pd.DataFrame:
    """Symmetric DTW distance matrix over the columns (stocks) of ``returns_window``."""
    cols = list(returns_window.columns)
    X = np.ascontiguousarray(returns_window.to_numpy().T, dtype=np.float64)  # (N, T)
    n = len(cols)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dtw_distance(X[i], X[j], band)
    return pd.DataFrame(D, index=cols, columns=cols)
