import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    _rebalance_positions,
    gate_passed_only_view,
    metrics_table,
    paired_block_bootstrap,
    performance_metrics,
    run_backtest,
)
from src.config import load_config
from src.data import align_to_common_days, load_fixture, to_returns

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
