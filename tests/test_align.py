from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.align import (
    distance_matrix,
    global_score,
    local_score,
    substitution_matrix,
)
from src.config import load_config

CFG = load_config().align


def _arr(xs):
    return np.array(xs, dtype=np.int64)


def test_substitution_matrix_graded_and_symmetric():
    S = substitution_matrix(2.0)
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 2.0)
    # adjacent symbols cost less than opposites
    assert S[4, 3] > S[4, 0]


def test_identical_strings_score_is_linear():
    S = substitution_matrix(2.0)
    a = _arr([4] * 5)
    assert global_score(a, a, S, CFG) == pytest.approx(10.0)


def test_hand_worked_small_case():
    S = substitution_matrix(2.0)
    # two matches plus one length-1 gap: 2 + 2 + (gap_open + gap_extend)
    score = global_score(_arr([4, 4]), _arr([4]), S, CFG)
    assert score == pytest.approx(2.0 + (CFG.gap_open + CFG.gap_extend))


def test_alignment_is_symmetric():
    S = substitution_matrix(2.0)
    a = _arr([0, 1, 2, 3, 4, 2])
    b = _arr([2, 3, 4, 0, 1])
    assert global_score(a, b, S, CFG) == pytest.approx(global_score(b, a, S, CFG))


def test_numba_matches_pure_python():
    S = substitution_matrix(2.0)
    py_cfg = replace(CFG, use_numba=False)
    rng = np.random.default_rng(1)
    for _ in range(8):
        a = rng.integers(0, 5, size=rng.integers(3, 20)).astype(np.int64)
        b = rng.integers(0, 5, size=rng.integers(3, 20)).astype(np.int64)
        assert global_score(a, b, S, CFG) == pytest.approx(global_score(a, b, S, py_cfg))


def test_band_limits_how_far_a_pattern_may_shift():
    S = substitution_matrix(2.0)
    a = _arr([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    b = np.roll(a, 3)
    assert global_score(a, b, S, CFG, band=1) < global_score(a, b, S, CFG, band=10)


def test_local_score_non_negative():
    S = substitution_matrix(2.0)
    a = _arr([0, 0, 0])
    b = _arr([4, 4, 4])
    assert local_score(a, b, S, CFG) >= 0.0


def _symbols(n_stocks=6, length=60, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 5, size=(length, n_stocks))
    idx = pd.date_range("2020-01-01", periods=length, freq="B")
    return pd.DataFrame(data, index=idx, columns=[f"S{i}" for i in range(n_stocks)])


def test_distance_matrix_symmetric_non_negative_zero_diagonal():
    sym = _symbols()
    D = distance_matrix(sym, CFG)
    A = D.to_numpy()
    assert np.allclose(A, A.T)
    assert (A >= -1e-12).all()
    assert np.allclose(np.diag(A), 0.0)
