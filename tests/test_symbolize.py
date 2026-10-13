import numpy as np
import pandas as pd
import pytest

from genomefolio.config import load_config
from genomefolio.data import align_to_common_days, load_fixture, to_returns
from genomefolio.symbolize import (
    BIG_DOWN,
    BIG_UP,
    FLAT,
    market_neutralize,
    quantile_edges,
    symbolize,
    symbolize_series,
)

CUTOFFS = (0.20, 0.40, 0.60, 0.80)


def _returns():
    cfg = load_config()
    return align_to_common_days(to_returns(load_fixture(cfg)))


def test_codes_within_alphabet():
    sym = symbolize(_returns(), CUTOFFS)
    assert sym.to_numpy().min() >= 0
    assert sym.to_numpy().max() <= 4


def test_symbols_are_roughly_balanced_per_stock():
    sym = symbolize(_returns(), CUTOFFS)
    shares = sym.apply(lambda c: c.value_counts(normalize=True).sort_index())
    # per-stock quantile binning -> each symbol ~20%; allow slack for ties
    assert (shares.min(axis=1) > 0.15).all()
    assert (shares.max(axis=1) < 0.25).all()


def test_constant_series_is_flat():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    out = symbolize_series(pd.Series([0.01] * 50, index=idx), CUTOFFS)
    assert (out == FLAT).all()


def test_nan_entries_get_nan_symbols():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    values = np.array([0.01, -0.02, np.nan, 0.03, -0.01, 0.0, 0.02, -0.03, 0.01, np.nan])
    out = symbolize_series(pd.Series(values, index=idx), CUTOFFS)
    assert out.isna().sum() == 2
    assert not out.dropna().isna().any()


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        quantile_edges(np.array([0.01, np.nan]), CUTOFFS)


def test_cutoff_change_moves_edges():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=500)
    e1 = quantile_edges(vals, (0.20, 0.40, 0.60, 0.80))
    e2 = quantile_edges(vals, (0.15, 0.35, 0.65, 0.85))
    assert not np.allclose(e1, e2)


def test_market_neutralize_sums_to_zero():
    r = _returns()
    mn = market_neutralize(r)
    assert np.allclose(mn.sum(axis=1), 0.0, atol=1e-12)


def test_market_neutral_option_changes_symbols():
    r = _returns()
    plain = symbolize(r, CUTOFFS, market_neutral=False)
    neutral = symbolize(r, CUTOFFS, market_neutral=True)
    assert not plain.equals(neutral)


def test_pooled_scope_preserves_magnitude():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    r = pd.DataFrame(
        {"low": rng.normal(0.0, 0.005, 500), "high": rng.normal(0.0, 0.03, 500)}, index=idx
    )
    per = symbolize(r, CUTOFFS, quantile_scope="per_stock")
    pooled = symbolize(r, CUTOFFS, quantile_scope="pooled")

    def extreme(s):
        return float(((s == BIG_DOWN) | (s == BIG_UP)).mean())

    # per-stock: both names have ~40% tail symbols -> magnitude ignored
    assert abs(extreme(per["low"]) - extreme(per["high"])) < 0.05
    # pooled: the high-volatility name draws far more extreme symbols
    assert extreme(pooled["high"]) > extreme(pooled["low"]) + 0.2


def test_invalid_quantile_scope_raises():
    with pytest.raises(ValueError):
        symbolize(_returns(), CUTOFFS, quantile_scope="bogus")


def test_symbolize_zscore_shape_range_and_warmup():
    from genomefolio.symbolize import symbolize_zscore

    r = _returns()
    z = symbolize_zscore(r, window=60)
    assert z.shape == r.shape
    finite = z.to_numpy()[~np.isnan(z.to_numpy())]
    assert finite.min() >= 0.0 and finite.max() <= 4.0
    assert z.iloc[:5].isna().any().any()  # warm-up NaNs at the start


def test_symbolize_zscore_has_no_look_ahead():
    from genomefolio.symbolize import symbolize_zscore

    r = _returns().iloc[-120:, :3]
    a = symbolize_zscore(r, window=30)
    r2 = r.copy()
    r2.iloc[-1] = r2.iloc[-1] + 10.0  # change only the most recent return
    b = symbolize_zscore(r2, window=30)
    assert a.iloc[:-1].equals(b.iloc[:-1])
