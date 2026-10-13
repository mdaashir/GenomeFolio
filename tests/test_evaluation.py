"""Tests for the evaluation helpers (deflated Sharpe, stationary bootstrap)."""

import numpy as np
import pandas as pd

from genomefolio.evaluation import (
    deflated_sharpe_from_returns,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    paired_stationary_bootstrap,
    probabilistic_sharpe_ratio,
    stationary_bootstrap_indices,
)


def test_psr_bounds_and_benchmark():
    assert probabilistic_sharpe_ratio(0.5, 252) > 0.99
    assert abs(probabilistic_sharpe_ratio(0.0, 252) - 0.5) < 1e-9


def test_dsr_shrinks_as_trials_grow():
    sr, n, skew, kurt = 0.1, 500, 0.0, 3.0
    b1 = expected_max_sharpe(1, 1.0 / n)
    b100 = expected_max_sharpe(100, 1.0 / n)
    assert b100 > b1
    d1 = deflated_sharpe_ratio(sr, n, skew, kurt, b1)
    d100 = deflated_sharpe_ratio(sr, n, skew, kurt, b100)
    assert d100 <= d1


def test_deflated_sharpe_from_returns_fields():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.001, 0.01, 300), index=idx)
    out = deflated_sharpe_from_returns(s, n_trials=10)
    assert set(out) == {"sharpe", "psr", "sr_benchmark", "dsr"}
    assert 0.0 <= out["psr"] <= 1.0
    assert 0.0 <= out["dsr"] <= 1.0


def test_stationary_bootstrap_indices_valid_and_deterministic():
    idx = stationary_bootstrap_indices(200, 0.2, np.random.default_rng(1))
    assert idx.shape == (200,)
    assert ((0 <= idx) & (idx < 200)).all()
    a = stationary_bootstrap_indices(200, 0.2, np.random.default_rng(5))
    b = stationary_bootstrap_indices(200, 0.2, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_paired_stationary_bootstrap_ordered():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    m = pd.Series(rng.normal(0.001, 0.01, 300), index=idx)
    b = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
    ci = paired_stationary_bootstrap(m, b, 200, 20, 0.95, rng)
    lo, hi = ci["sharpe_diff"]
    assert lo <= hi
