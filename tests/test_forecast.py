from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.forecast import forecast_clusters, forecast_series

CFG = load_config().forecast


def _series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0003, 0.01, n), index=idx)


def test_arima_forecast_is_finite_and_valid_order():
    fc = forecast_series(_series(), CFG, 21)
    assert np.isfinite(fc.expected_return)
    if fc.order is not None:
        p, d, q = fc.order
        assert 0 <= p <= CFG.max_p
        assert 0 <= q <= CFG.max_q
        assert d in (0, 1)


def test_short_series_falls_back_to_mean():
    fc = forecast_series(_series(n=10), CFG, 21)
    assert fc.fallback
    assert fc.model == "historical_mean"


def test_constant_series_falls_back_to_zero():
    fc = forecast_series(pd.Series([0.0] * 100), CFG, 21)
    assert fc.expected_return == pytest.approx(0.0)


def test_forecast_clusters_covers_all_columns():
    df = pd.DataFrame({0: _series(seed=1).to_numpy(), 1: _series(seed=2).to_numpy()})
    out = forecast_clusters(df, CFG, 21)
    assert set(out) == {0, 1}


def test_spline_mode_is_finite():
    cfg = replace(CFG, model="spline")
    fc = forecast_series(_series(), cfg, 21)
    assert np.isfinite(fc.expected_return)
    assert fc.model == "spline"
