import numpy as np
import pytest

from src.config import load_config, load_sectors
from src.data import align_to_common_days, load_fixture, to_returns
from src.portfolio import (
    cap_redistribute,
    correlation_clusters,
    equal_weights,
    optimised_weights,
    random_clusters,
    sector_clusters,
    two_stage_weights,
)

CFG = load_config()


def _returns_window(n=10, days=252):
    r = align_to_common_days(to_returns(load_fixture(CFG)))
    return r.iloc[-days:, :n]


def test_cap_redistribute_preserves_total_and_cap():
    out = cap_redistribute(np.array([0.9, 0.05, 0.05]), 0.4)
    assert out.sum() == pytest.approx(1.0)
    assert (out <= 0.4 + 1e-9).all()


def test_cap_redistribute_infeasible_raises():
    # 8 x 10% = 80% cannot hold a fully invested portfolio
    with pytest.raises(ValueError):
        cap_redistribute(np.full(8, 1.0 / 8.0), 0.10)


def test_cap_redistribute_with_array_caps():
    out = cap_redistribute(np.array([0.5, 0.25, 0.25]), np.array([0.2, 0.5, 0.5]))
    assert out.sum() == pytest.approx(1.0)
    assert out[0] <= 0.2 + 1e-9


def test_two_stage_weights_valid_and_capped():
    win = _returns_window()
    clusters = {0: tuple(win.columns[:4]), 1: tuple(win.columns[4:])}
    w = two_stage_weights(clusters, win, CFG.portfolio)
    cap = CFG.portfolio.cap(len(win.columns))
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-12).all()
    assert (w <= cap + 1e-9).all()


def test_two_stage_eight_stock_feasible():
    win = _returns_window(n=8)
    clusters = {0: tuple(win.columns[:4]), 1: tuple(win.columns[4:])}
    w = two_stage_weights(clusters, win, CFG.portfolio)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_two_stage_forecast_tilt_still_valid():
    win = _returns_window()
    clusters = {0: tuple(win.columns[:4]), 1: tuple(win.columns[4:])}
    from dataclasses import replace

    cfg = replace(CFG.portfolio, cluster_method="forecast_tilt")
    w = two_stage_weights(clusters, win, cfg, forecasts={0: 0.01, 1: -0.01})
    assert w.sum() == pytest.approx(1.0)


def test_equal_weights():
    w = equal_weights(["a", "b", "c"])
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w.to_numpy(), 1.0 / 3.0)


def test_optimised_weights_respect_cap_and_sum():
    win = _returns_window()
    cap = CFG.portfolio.cap(len(win.columns))
    w = optimised_weights(win, cap)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()
    assert (w <= cap + 1e-6).all()


def test_random_clusters_sizes_match():
    cl = random_clusters([3, 5], list("abcdefgh"), np.random.default_rng(0))
    assert sorted(len(v) for v in cl.values()) == [3, 5]
    assert sorted(t for v in cl.values() for t in v) == sorted("abcdefgh")


def test_correlation_clusters_covers_all():
    win = _returns_window()
    cl = correlation_clusters(win, CFG.cluster)
    assert sorted(t for v in cl.values() for t in v) == sorted(win.columns)


def test_sector_clusters_restricted_to_universe():
    sectors = load_sectors()
    tickers = list(CFG.universe.tickers)
    cl = sector_clusters(sectors, tickers)
    assert sorted(t for v in cl.values() for t in v) == sorted(tickers)
