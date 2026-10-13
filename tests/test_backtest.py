import numpy as np
import pandas as pd
import pytest

from genomefolio.backtest import (
    _rebalance_positions,
    gate_passed_only_view,
    metrics_table,
    paired_block_bootstrap,
    performance_metrics,
    politis_white_block_length,
    run_backtest,
    seed_for,
)
from genomefolio.config import load_config
from genomefolio.data import align_to_common_days, load_fixture, to_returns

CFG = load_config()


def _returns(days=400, n=10):
    r = align_to_common_days(to_returns(load_fixture(CFG)))
    return r.iloc[-days:, :n]


def test_performance_metrics_constant_positive():
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    daily = pd.Series(0.001, index=dates)
    m = performance_metrics(daily)
    assert m["ann_vol"] == pytest.approx(0.0)
    assert np.isnan(m["sharpe"])  # zero variance -> undefined Sharpe
    assert m["max_drawdown"] == pytest.approx(0.0)


def test_rebalance_positions():
    pos = _rebalance_positions(1000, 252, 21)
    assert pos[0] == 252
    assert pos[1] == 273
    assert all(p + 21 <= 1000 for p in pos)


def test_run_backtest_equal_weight():
    res = run_backtest(_returns(), CFG, method="equal_weight", cost_bps=0.0)
    assert len(res.daily_returns) > 0
    # first period trades from all-zero to a fully invested book -> turnover = 1
    assert list(res.turnover)[0] == pytest.approx(1.0)


def test_run_backtest_alignment_gate_fraction_bounded():
    res = run_backtest(_returns(), CFG, method="alignment", n_draws=5)
    assert len(res.daily_returns) > 0
    assert 0.0 <= res.gate_pass_fraction <= 1.0


def test_all_methods_book_fully_invested():
    r = _returns()
    for method in ("equal_weight", "correlation", "sector", "random"):
        res = run_backtest(r, CFG, method=method, n_draws=5, cost_bps=0.0)
        for date, w in res.weights.items():
            assert abs(float(w.sum()) - 1.0) < 1e-6, f"{method} under-invested on {date}"


def test_sector_method_uncovered_universe_raises():
    r = _returns().copy()
    r.columns = [f"ZZZ{i}" for i in range(r.shape[1])]
    with pytest.raises(ValueError, match="does not cover"):
        run_backtest(r, CFG, method="sector", n_draws=5)


def test_no_look_ahead_future_perturbation():
    r = _returns(days=400)
    res1 = run_backtest(r, CFG, method="alignment", n_draws=5)
    dates = list(res1.weights)
    first = dates[0]

    r2 = r.copy()
    pos = r.index.get_loc(first)
    r2.iloc[pos:] = -r2.iloc[pos:]  # flip every future return (changes the pattern)

    res2 = run_backtest(r2, CFG, method="alignment", n_draws=5)
    # the first rebalance saw only data before `first`, so its weights must be identical
    assert np.allclose(res1.weights[first].to_numpy(), res2.weights[first].to_numpy())
    # later rebalances do see the perturbed data, so at least one must differ
    assert any(
        not np.allclose(res1.weights[d].to_numpy(), res2.weights[d].to_numpy()) for d in dates[1:]
    )


def test_run_backtest_is_reproducible():
    r = _returns()
    a = run_backtest(r, CFG, method="alignment", n_draws=5)
    b = run_backtest(r, CFG, method="alignment", n_draws=5)
    assert a.daily_returns.equals(b.daily_returns)
    assert a.turnover.equals(b.turnover)
    for d in a.weights:
        assert a.weights[d].equals(b.weights[d])


def test_cost_reduces_return():
    free = run_backtest(_returns(), CFG, method="equal_weight", cost_bps=0.0)
    costly = run_backtest(_returns(), CFG, method="equal_weight", cost_bps=25.0)
    assert costly.daily_returns.sum() <= free.daily_returns.sum() + 1e-12


def test_paired_block_bootstrap_ci_ordered():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    method = pd.Series(rng.normal(0.001, 0.01, 300), index=idx)
    base = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
    ci = paired_block_bootstrap(method, base, 200, 20, 0.95, rng)
    for key in ("sharpe_method", "sharpe_baseline", "sharpe_diff", "return_diff"):
        lo, hi = ci[key]
        assert lo <= hi


def test_metrics_table_and_descriptive_view():
    results = {
        m: run_backtest(_returns(), CFG, method=m, n_draws=5, cost_bps=0.0)
        for m in ("alignment", "equal_weight", "sector")
    }
    table = metrics_table(results)
    assert set(table.index) == {"alignment", "equal_weight", "sector"}
    assert "gate_pass_fraction" in table.columns
    desc = gate_passed_only_view(results, reference="alignment")
    assert set(desc.index) == set(table.index)


def test_politis_white_block_length_in_range():
    rng = np.random.default_rng(0)
    b = politis_white_block_length(rng.normal(size=1000))
    assert 1 <= b <= 1000


def test_politis_white_grows_with_autocorrelation():
    rng = np.random.default_rng(1)
    n = 2000
    white = rng.normal(size=n)
    eps = rng.normal(size=n)
    ar = np.zeros(n)
    ar[0] = eps[0]
    for t in range(1, n):
        ar[t] = 0.8 * ar[t - 1] + eps[t]
    assert politis_white_block_length(ar) > politis_white_block_length(white)


def test_politis_white_short_and_degenerate_series():
    assert 1 <= politis_white_block_length(np.array([0.01, -0.02, 0.03])) <= 3
    assert 1 <= politis_white_block_length(np.zeros(500)) <= 500


def test_seed_for_is_deterministic():
    assert seed_for("alignment", 42) == seed_for("alignment", 42)
    assert seed_for("alignment", 42) != seed_for("equal_weight", 42)


def test_bootstrap_cis_are_order_independent():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    series = {
        "x": pd.Series(rng.normal(0.001, 0.01, 300), index=idx),
        "y": pd.Series(rng.normal(0.0005, 0.01, 300), index=idx),
    }
    base = pd.Series(rng.normal(0.0002, 0.01, 300), index=idx)

    def compute(order):
        return {
            name: paired_block_bootstrap(
                series[name], base, 300, 20, 0.95, np.random.default_rng(seed_for(name, 42))
            )
            for name in order
        }

    a, b = compute(["x", "y"]), compute(["y", "x"])
    assert a["x"]["sharpe_diff"] == b["x"]["sharpe_diff"]
    assert a["y"]["sharpe_diff"] == b["y"]["sharpe_diff"]


def test_alt_fixture_sector_control_is_fully_invested():
    from dataclasses import replace

    cfg = replace(
        CFG,
        universe=replace(
            CFG.universe,
            tickers=("ORCL", "ADBE", "CRM", "QCOM", "TXN", "IBM", "ABT", "BMY", "LLY", "TGT", "LOW", "SBUX"),
            fixture_path="data/sample/prices_alt.csv",
            sectors_path="config/sectors_alt.yaml",
        ),
    )
    r = align_to_common_days(to_returns(load_fixture(cfg)))
    assert r.shape[1] == 12
    res = run_backtest(r, cfg, method="sector", n_draws=5, cost_bps=0.0)
    for w in res.weights.values():
        assert abs(float(w.sum()) - 1.0) < 1e-6
