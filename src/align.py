"""Pairwise sequence alignment of symbol strings.

Global alignment (Needleman-Wunsch) with **affine gaps** and a **band** that caps how
far a gap may shift an index. Smith-Waterman (local) is provided only as a flagged
cross-check.

Scoring
-------
Symbols are ordinals 0..4 (big-down .. big-up). The substitution matrix is graded:

    score(s, t) = match_score            if s == t
                = -|s - t|               otherwise

so adjacent symbols (e.g. small-up vs big-up) cost less than opposites, and the matrix
is symmetric by construction.

Distance
--------
Raw scores are normalised by the two strings' self-scores:

    similarity(a, b) = score(a, b) / sqrt(self(a) * self(b))
    distance(a, b)   = 1 - similarity(a, b)

This is **not claimed to be a metric** (it may violate the triangle inequality); the
distance matrix is asserted symmetric and non-negative only, which is all average
linkage needs. Smith-Waterman local scores are even less metric-like, which is why they
are a cross-check and not the clustering input.

No look-ahead: alignment operates on whatever strings it is given; callers pass the
current training window only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AlignConfig

NEG = -1.0e18

try:  # pragma: no cover - exercised implicitly by the installed environment
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn

        return wrap


def substitution_matrix(match_score: float, n_symbols: int = 5) -> np.ndarray:
    """Graded 5x5 substitution matrix (symmetric)."""
    s = np.empty((n_symbols, n_symbols), dtype=np.float64)
    for i in range(n_symbols):
        for j in range(n_symbols):
            s[i, j] = match_score if i == j else -abs(i - j)
    return s


@njit(cache=True)
def _nw_banded_numba(a, b, S, gap_open, gap_extend, band):  # pragma: no cover
    n = a.shape[0]
    m = b.shape[0]
    M = np.full((n + 1, m + 1), NEG)
    X = np.full((n + 1, m + 1), NEG)
    Y = np.full((n + 1, m + 1), NEG)
    M[0, 0] = 0.0
    for i in range(1, n + 1):
        if i <= band:
            X[i, 0] = gap_open + i * gap_extend
    for j in range(1, m + 1):
        if j <= band:
            Y[0, j] = gap_open + j * gap_extend
    for i in range(1, n + 1):
        jlo = i - band
        if jlo < 1:
            jlo = 1
        jhi = i + band
        if jhi > m:
            jhi = m
        for j in range(jlo, jhi + 1):
            best = M[i - 1, j - 1]
            if X[i - 1, j - 1] > best:
                best = X[i - 1, j - 1]
            if Y[i - 1, j - 1] > best:
                best = Y[i - 1, j - 1]
            M[i, j] = best + S[a[i - 1], b[j - 1]]

            x1 = M[i - 1, j] + gap_open + gap_extend
            x2 = X[i - 1, j] + gap_extend
            X[i, j] = x1 if x1 > x2 else x2

            y1 = M[i, j - 1] + gap_open + gap_extend
            y2 = Y[i, j - 1] + gap_extend
            Y[i, j] = y1 if y1 > y2 else y2
    out = M[n, m]
    if X[n, m] > out:
        out = X[n, m]
    if Y[n, m] > out:
        out = Y[n, m]
    return out


def _nw_banded_py(a, b, S, gap_open, gap_extend, band):
    n, m = len(a), len(b)
    M = np.full((n + 1, m + 1), NEG)
    X = np.full((n + 1, m + 1), NEG)
    Y = np.full((n + 1, m + 1), NEG)
    M[0, 0] = 0.0
    for i in range(1, n + 1):
        if i <= band:
            X[i, 0] = gap_open + i * gap_extend
    for j in range(1, m + 1):
        if j <= band:
            Y[0, j] = gap_open + j * gap_extend
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            M[i, j] = max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1]) + S[a[i - 1], b[j - 1]]
            X[i, j] = max(M[i - 1, j] + gap_open + gap_extend, X[i - 1, j] + gap_extend)
            Y[i, j] = max(M[i, j - 1] + gap_open + gap_extend, Y[i, j - 1] + gap_extend)
    return max(M[n, m], X[n, m], Y[n, m])


@njit(cache=True)
def _sw_numba(a, b, S, gap_open, gap_extend):  # pragma: no cover
    n = a.shape[0]
    m = b.shape[0]
    M = np.zeros((n + 1, m + 1))
    X = np.full((n + 1, m + 1), NEG)
    Y = np.full((n + 1, m + 1), NEG)
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            m_prev = M[i - 1, j - 1]
            x_prev = X[i - 1, j - 1]
            y_prev = Y[i - 1, j - 1]
            prev = m_prev
            if x_prev > prev:
                prev = x_prev
            if y_prev > prev:
                prev = y_prev
            val = prev + S[a[i - 1], b[j - 1]]
            M[i, j] = val if val > 0.0 else 0.0
            x1 = M[i - 1, j] + gap_open + gap_extend
            x2 = X[i - 1, j] + gap_extend
            xv = x1 if x1 > x2 else x2
            X[i, j] = xv if xv > 0.0 else 0.0
            y1 = M[i, j - 1] + gap_open + gap_extend
            y2 = Y[i, j - 1] + gap_extend
            yv = y1 if y1 > y2 else y2
            Y[i, j] = yv if yv > 0.0 else 0.0
            if M[i, j] > best:
                best = M[i, j]
            if X[i, j] > best:
                best = X[i, j]
            if Y[i, j] > best:
                best = Y[i, j]
    return best


def _sw_py(a, b, S, gap_open, gap_extend):
    n, m = len(a), len(b)
    M = np.zeros((n + 1, m + 1))
    X = np.full((n + 1, m + 1), NEG)
    Y = np.full((n + 1, m + 1), NEG)
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            val = max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1]) + S[a[i - 1], b[j - 1]]
            M[i, j] = max(val, 0.0)
            X[i, j] = max(M[i - 1, j] + gap_open + gap_extend, X[i - 1, j] + gap_extend, 0.0)
            Y[i, j] = max(M[i, j - 1] + gap_open + gap_extend, Y[i, j - 1] + gap_extend, 0.0)
            best = max(best, M[i, j], X[i, j], Y[i, j])
    return best


def global_score(a: np.ndarray, b: np.ndarray, S: np.ndarray, cfg: AlignConfig, band: int | None = None) -> float:
    """Banded global (Needleman-Wunsch) alignment score."""
    band = cfg.band if band is None else band
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    if _HAVE_NUMBA and cfg.use_numba:
        return float(_nw_banded_numba(a, b, S, cfg.gap_open, cfg.gap_extend, band))
    return float(_nw_banded_py(a, b, S, cfg.gap_open, cfg.gap_extend, band))


def local_score(a: np.ndarray, b: np.ndarray, S: np.ndarray, cfg: AlignConfig) -> float:
    """Smith-Waterman local score (cross-check only, not used for clustering)."""
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    if _HAVE_NUMBA and cfg.use_numba:
        return float(_sw_numba(a, b, S, cfg.gap_open, cfg.gap_extend))
    return float(_sw_py(a, b, S, cfg.gap_open, cfg.gap_extend))


def _as_codes(series: pd.Series) -> np.ndarray:
    return np.asarray(series.to_numpy(), dtype=np.int64)


def distance_matrix(
    symbols: pd.DataFrame, cfg: AlignConfig, band: int | None = None
) -> pd.DataFrame:
    """Normalised alignment-distance matrix for the stocks in ``symbols``.

    Symmetric with a zero diagonal. Asserts (and relies on) symmetry and
    non-negativity; it is not asserted to be a metric.
    """
    cols = list(symbols.columns)
    codes = {c: _as_codes(symbols[c]) for c in cols}
    S = substitution_matrix(cfg.match_score)
    self_scores = {c: global_score(codes[c], codes[c], S, cfg, band) for c in cols}

    n = len(cols)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cols[i], cols[j]
            s = global_score(codes[a], codes[b], S, cfg, band)
            denom = np.sqrt(self_scores[a] * self_scores[b])
            sim = s / denom if denom > 0 else 0.0
            d = 1.0 - sim
            if d < 0.0:
                d = 0.0
            D[i, j] = D[j, i] = d

    if not np.allclose(D, D.T, atol=1e-9):
        raise AssertionError("distance matrix is not symmetric")
    if np.any(D < -1e-9):
        raise AssertionError("distance matrix has negative entries")
    return pd.DataFrame(D, index=cols, columns=cols)
