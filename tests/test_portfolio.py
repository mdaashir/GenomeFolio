import numpy as np
import pandas as pd
import pytest

from genomefolio.config import load_config, load_sectors
from genomefolio.data import align_to_common_days, load_fixture, to_returns
from genomefolio.portfolio import (
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


def test_equal_weights_empty_raises():
    with pytest.raises(ValueError):
        equal_weights([])


def test_two_stage_empty_clusters_raises():
    win = _returns_window()
    with pytest.raises(ValueError):
        two_stage_weights({}, win, CFG.portfolio)


def test_optimised_weights_respect_cap_and_sum():
    win = _returns_window()
    cap = CFG.portfolio.cap(len(win.columns))
    w = optimised_weights(win, cap)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()
    assert (w <= cap + 1e-6).all()


def test_mean_shrink_only_affects_mean_variance():
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 0.01, (252, 4))
    X[:, 0] += 0.002  # a stock with a strong positive mean
    win = pd.DataFrame(X, index=idx, columns=list("abcd"))
    cap = CFG.portfolio.cap(4)
    # minimum-variance ignores means, so shrinkage must not change it
    mv0 = optimised_weights(win, cap, risk_aversion=0.0, mean_shrink=0.0)
    mv1 = optimised_weights(win, cap, risk_aversion=0.0, mean_shrink=0.9)
    assert np.allclose(mv0.to_numpy(), mv1.to_numpy())
    # mean-variance tilts toward the high-mean stock; shrinking tempers it
    full = optimised_weights(win, cap, risk_aversion=5.0, mean_shrink=0.0)
    shrunk = optimised_weights(win, cap, risk_aversion=5.0, mean_shrink=0.99)
    assert full["a"] >= shrunk["a"] - 1e-9


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


def test_risk_parity_honored_without_forecasts():
    from dataclasses import replace

    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    rng = np.random.default_rng(0)
    low = rng.normal(0.0, 0.005, (252, 2))
    high = rng.normal(0.0, 0.03, (252, 2))
    win = pd.DataFrame(np.hstack([low, high]), index=idx, columns=list("abcd"))
    clusters = {0: ("a", "b"), 1: ("c", "d")}
    cfg_eq = replace(CFG.portfolio, cluster_method="equal")
    cfg_rp = replace(CFG.portfolio, cluster_method="risk_parity")
    w_eq = two_stage_weights(clusters, win, cfg_eq, forecasts=None)
    w_rp = two_stage_weights(clusters, win, cfg_rp, forecasts=None)
    assert not np.allclose(w_eq.to_numpy(), w_rp.to_numpy())
    # the low-volatility cluster should receive more total weight
    assert w_rp[["a", "b"]].sum() > w_rp[["c", "d"]].sum()


def test_forecast_tilt_without_forecasts_falls_back_to_equal():
    from dataclasses import replace

    win = _returns_window()
    clusters = {0: tuple(win.columns[:5]), 1: tuple(win.columns[5:])}
    cfg_tilt = replace(CFG.portfolio, cluster_method="forecast_tilt")
    w = two_stage_weights(clusters, win, cfg_tilt, forecasts=None)
    assert w.sum() == pytest.approx(1.0)


def test_optimised_weights_handles_degenerate_covariance():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    win = pd.DataFrame(np.zeros((100, 4)), index=idx, columns=list("abcd"))
    cap = CFG.portfolio.cap(4)
    w = optimised_weights(win, cap)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()
    assert (w <= cap + 1e-6).all()


def test_inverse_vol_handles_zero_variance_stock():
    from dataclasses import replace

    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 0.01, (100, 4))
    X[:, 3] = 0.0  # a zero-variance stock
    win = pd.DataFrame(X, index=idx, columns=list("abcd"))
    cfg = replace(CFG.portfolio, within_cluster="inverse_vol")
    w = two_stage_weights({0: ("a", "b"), 1: ("c", "d")}, win, cfg)
    assert w.sum() == pytest.approx(1.0)
    assert np.isfinite(w.to_numpy()).all()


def test_cap_non_binding_leaves_weights_unsaturated():
    win = _returns_window()
    clusters = {0: tuple(win.columns[:5]), 1: tuple(win.columns[5:])}
    from dataclasses import replace

    cfg = replace(CFG.portfolio, cap_floor=1.0)  # cap never binds with 10 stocks
    w = two_stage_weights(clusters, win, cfg)
    assert w.sum() == pytest.approx(1.0)
    assert (w < 1.0).all()  # no single stock held whole


def test_ledoit_wolf_allocators_are_valid_and_differ_from_sample():
    from dataclasses import replace

    win = _returns_window()
    clusters = {0: tuple(win.columns[:4]), 1: tuple(win.columns[4:])}
    for cm, wc in (("risk_parity_lw", "inverse_vol_lw"), ("risk_parity", "inverse_vol")):
        cfg = replace(CFG.portfolio, cluster_method=cm, within_cluster=wc)
        w = two_stage_weights(clusters, win, cfg)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-12).all()
    sample = two_stage_weights(clusters, win, replace(CFG.portfolio, cluster_method="risk_parity"))
    lw = two_stage_weights(clusters, win, replace(CFG.portfolio, cluster_method="risk_parity_lw"))
    assert not np.allclose(sample.to_numpy(), lw.to_numpy())
