"""Tests for the DTW baseline."""

import numpy as np
import pandas as pd
import pytest

from genomefolio.dtw import dtw_distance, dtw_distance_matrix


def test_dtw_identical_is_zero_and_symmetric():
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    b = rng.normal(size=50)
    assert dtw_distance(a, a) == pytest.approx(0.0)
    assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a))


def test_dtw_band_guard():
    with pytest.raises(ValueError):
        dtw_distance(np.zeros(5), np.zeros(2), band=0)


def test_dtw_distance_matrix_valid():
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(1)
    win = pd.DataFrame(rng.normal(size=(60, 5)), index=idx, columns=list("abcde"))
    D = dtw_distance_matrix(win, band=5)
    A = D.to_numpy()
    assert np.allclose(A, A.T)
    assert (A >= 0).all()
    assert np.allclose(np.diag(A), 0.0)


def test_dtw_similar_series_are_closer():
    rng = np.random.default_rng(2)
    base = rng.normal(size=80)
    similar = base + 0.01 * rng.normal(size=80)
    different = rng.normal(size=80)
    assert dtw_distance(base, similar, band=5) < dtw_distance(base, different, band=5)
