"""Pipeline tests: symbolise -> align -> cluster -> forecast (no gate, no portfolio)."""

import numpy as np

from src.config import load_config
from src.data import align_to_common_days, load_fixture, to_returns
from src.pipeline import run_pipeline

CFG = load_config()


def _window():
    returns = align_to_common_days(to_returns(load_fixture(CFG)))
    return returns.iloc[-CFG.window.train_days :]


def test_pipeline_clusters_every_stock_and_forecasts_each_cluster():
    window = _window()
    result = run_pipeline(window, CFG)
    assert result.n_clusters >= 1
    assert set(result.clusters) == set(result.forecasts)
    members = [t for group in result.clusters.values() for t in group]
    assert sorted(members) == sorted(window.columns)
    for forecast in result.forecasts.values():
        assert np.isfinite(forecast.expected_return)


def test_pipeline_is_deterministic():
    window = _window()
    a = run_pipeline(window, CFG)
    b = run_pipeline(window, CFG)
    assert a.clusters == b.clusters
    assert {c: f.expected_return for c, f in a.forecasts.items()} == {
        c: f.expected_return for c, f in b.forecasts.items()
    }
