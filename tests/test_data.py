from src.config import load_config
from src.data import (
    align_to_common_days,
    load_fixture,
    time_split,
    to_returns,
)


def _returns():
    cfg = load_config()
    return align_to_common_days(to_returns(load_fixture(cfg)))


def test_fixture_size_and_history():
    cfg = load_config()
    prices = load_fixture(cfg)
    assert prices.shape[1] >= 8, "fixture must have >= 8 stocks"
    years = (prices.index.max() - prices.index.min()).days / 365.25
    assert years >= 3, "fixture must span >= 3 years"
    # enough history for a 252-day window plus several monthly rebalances
    assert len(prices) >= cfg.window.train_days + 6 * cfg.window.rebalance_days


def test_returns_have_no_nan_after_alignment():
    r = _returns()
    assert not r.isna().any().any()


def test_time_split_disabled_returns_empty_report():
    r = _returns()
    tune, report = time_split(r, 0.7, enabled=False)
    assert len(tune) == len(r)
    assert report.empty


def test_time_split_enabled_is_chronological():
    r = _returns()
    tune, report = time_split(r, 0.7, enabled=True)
    assert len(tune) + len(report) == len(r)
    assert tune.index.max() < report.index.min()
    assert len(tune) == int(len(r) * 0.7)
